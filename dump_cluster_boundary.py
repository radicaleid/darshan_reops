import ROOT
import sys
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# Optimized branch processor
# -----------------------------
class BranchBasketBytes:
    def __init__(self, branch: ROOT.TBranch):
        max_b = branch.GetMaxBaskets()
        self.basketFirstEntry = np.frombuffer(branch.GetBasketEntry(),
                                              dtype=np.int64, count=max_b)
        self.basketBytes = np.frombuffer(branch.GetBasketBytes(),
                                         dtype=np.int32, count=max_b)
        self.branchName = branch.GetName()
        self.maxBaskets = max_b
        self.iBasket = 0
        self.isAligned = True

    def isAlignedWithClusterBoundaries(self):
        return self.isAligned

    def bytesInNextCluster(self, clusterBegin, clusterEnd):
        """Optimized NumPy + searchsorted (handles non-event trees correctly)"""

        if self.iBasket >= self.maxBaskets:
            self.isAligned = False
            return (0, 0, True)

        # Allow basketFirstEntry <= clusterBegin (baseline permits this)
        if self.basketFirstEntry[self.iBasket] > clusterBegin:
            # basket starts after cluster begin → misaligned
            self.isAligned = False
            return (0, 0, True)

        start = self.iBasket
        end   = self.maxBaskets

        # Find how many baskets have entry < clusterEnd
        rel_end_idx = np.searchsorted(self.basketFirstEntry[start:end],
                                      clusterEnd, side='left')
        nbaskets = int(rel_end_idx)

        if nbaskets <= 0:
            self.isAligned = False
            return (0, 0, True)

        bytes_ = int(self.basketBytes[start:start + nbaskets].sum())
        self.iBasket += nbaskets

        # Alignment check at end boundary
        if self.iBasket >= self.maxBaskets or self.basketFirstEntry[self.iBasket] != clusterEnd:
            self.isAligned = False
            return (0, 0, True)

        return (bytes_, nbaskets, False)


# -----------------------------
# Reference baseline processor
# -----------------------------
class BranchBasketBytesBaseline:
    def __init__(self, branch: ROOT.TBranch):
        self.basketFirstEntry = branch.GetBasketEntry()
        self.basketBytes = branch.GetBasketBytes()
        self.branchName = branch.GetName()
        self.maxBaskets = branch.GetMaxBaskets()
        self.iBasket = 0
        self.isAligned = True

    def isAlignedWithClusterBoundaries(self):
        return self.isAligned

    def bytesInNextCluster(self, clusterBegin, clusterEnd):
        if self.basketFirstEntry[self.iBasket] > clusterBegin:
            self.isAligned = False
            return (0, 0, True)

        bytes_ = 0
        nbaskets = 0
        while self.iBasket < self.maxBaskets and self.basketFirstEntry[self.iBasket] < clusterEnd:
            bytes_ += self.basketBytes[self.iBasket]
            nbaskets += 1
            self.iBasket += 1

        if self.iBasket >= self.maxBaskets or self.basketFirstEntry[self.iBasket] != clusterEnd:
            self.isAligned = False
            return (0, 0, True)

        return (bytes_, nbaskets, False)


# -----------------------------
# Helpers
# -----------------------------
def makeProcessors(branch, isEventsTree, cls):
    """Recursive branch traversal, return processors of type `cls`."""
    ret = []
    subBranches = branch.GetListOfBranches()
    if subBranches and subBranches.GetEntries() > 0:
        for i in range(subBranches.GetEntries()):
            ret.extend(makeProcessors(subBranches.At(i), isEventsTree, cls))
    else:
        ret.append(cls(branch))
    return ret


def clusterPrint(tr, isEventsTree, max_workers=None, validate=False, per_branch=False):
    clusterIter = tr.GetClusterIterator(0)
    nentries = tr.GetEntries()

    # Build processors
    processors = []
    branches = tr.GetListOfBranches()
    for i in range(branches.GetEntries()):
        processors.extend(makeProcessors(branches.At(i), isEventsTree, BranchBasketBytes))

    if validate:
        processors_ref = []
        for i in range(branches.GetEntries()):
            processors_ref.extend(makeProcessors(branches.At(i), isEventsTree, BranchBasketBytesBaseline))
    else:
        processors_ref = None

    if not per_branch:
        print(f"{'Begin':>15}{'End':>15}{'Entries':>15}"
              f"{'Max baskets':>15}{'Start byte':>15}{'Bytes':>15}")
    else:
        header = (f"{'Cluster':>8}{'Branch':>170}{'Begin':>10}{'End':>10}"
                  f"{'Entries':>8}{'Start':>15}{'End':>15}"
                  f"{'Bytes':>15}{'Baskets':>8}")
        print(header)

    nonAligned = set()
    nonAlignedRef = set()
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1))

    clusterBegin = clusterIter()
    start_byte = 0  # global offset

    # track per-branch totals
    branch_totals = {p.branchName: {"entries": 0, "bytes": 0, "baskets": 0,
                                    "start": None, "end": None}
                     for p in processors}

    cluster_idx = 0
    while clusterBegin < nentries:
        clusterEnd = clusterIter.GetNextEntry()
        bytes_total = 0
        maxbaskets = 0
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(p.bytesInNextCluster, clusterBegin, clusterEnd): p
                       for p in processors if p.isAlignedWithClusterBoundaries()}
            for fut in futures:
                bytes_, nbaskets, misaligned = fut.result()
                pname = futures[fut].branchName
                results.append((pname, bytes_, nbaskets, misaligned))
                bytes_total += bytes_
                maxbaskets = max(maxbaskets, nbaskets)
                if misaligned:
                    nonAligned.add(pname)

        if not per_branch:
            # global print
            print(f"{clusterBegin:15}{clusterEnd:15}{clusterEnd - clusterBegin:15}"
                  f"{maxbaskets:15}{start_byte:15}{bytes_total:15}")
            start_byte += bytes_total
        else:
            # per-branch print with global start & inclusive end
            entries = clusterEnd - clusterBegin
            cluster_start = start_byte

            for branch, bytes_, nbaskets, misaligned in results:
                if bytes_ > 0:
                    branch_end = cluster_start + bytes_ - 1  # inclusive end
                else:
                    branch_end = cluster_start - 1  # empty branch

                print(f"{cluster_idx:8}{branch:>170}{clusterBegin:10}{clusterEnd:10}"
                      f"{entries:8}{cluster_start:15}{branch_end:15}"
                      f"{bytes_:15}{nbaskets:8}")

                # update totals
                bt = branch_totals[branch]
                bt["entries"] += entries
                bt["bytes"]   += bytes_
                bt["baskets"] += nbaskets
                if bt["start"] is None:
                    bt["start"] = cluster_start
                bt["end"] = branch_end

                cluster_start = branch_end + 1  # advance

            start_byte += bytes_total

        # Validation
        if validate:
            bytes_total_ref = 0
            maxbaskets_ref = 0
            for p in processors_ref:
                if p.isAlignedWithClusterBoundaries():
                    byt, bas, mis = p.bytesInNextCluster(clusterBegin, clusterEnd)
                    bytes_total_ref += byt
                    maxbaskets_ref = max(maxbaskets_ref, bas)
                    if mis:
                        nonAlignedRef.add(p.branchName)

            if (bytes_total_ref != bytes_total) or (maxbaskets_ref != maxbaskets):
                print(f"❌ Validation mismatch at cluster {clusterBegin}-{clusterEnd}: "
                      f"optimized=({bytes_total},{maxbaskets}) vs "
                      f"baseline=({bytes_total_ref},{maxbaskets_ref})")
                sys.exit(1)

        cluster_idx += 1
        clusterBegin = clusterIter()

    # if per_branch:
    #     # --- Per-branch summary ---
    #     print("\nPer-branch totals:")
    #     header = (f"{'Branch':>70}{'Entries':>15}{'Bytes':>15}"
    #               f"{'Baskets':>15}{'Start':>15}{'End':>15}")
    #     print(header)

    #     for branch, stats in branch_totals.items():
    #         print(f"{branch:>70}{stats['entries']:15}{stats['bytes']:15}"
    #               f"{stats['baskets']:15}{stats['start']:15}{stats['end']:15}")

    if validate:
        if nonAligned != nonAlignedRef:
            print("❌ Validation mismatch in nonAlignedBranches sets")
            print("Optimized only:", nonAligned - nonAlignedRef)
            print("Baseline only:", nonAlignedRef - nonAligned)
            sys.exit(1)
        else:
            print("✅ Validation passed: optimized matches baseline.")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} input.root tree_name [isEventsTree=0/1] [max_workers] [validate=0/1] [per_branch]=0/1")
        sys.exit(1)

    input_file = sys.argv[1]
    tree_name = sys.argv[2]
    isEventsTree = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False
    max_workers = int(sys.argv[4]) if len(sys.argv) > 4 else None
    validate = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False
    per_branch = bool(int(sys.argv[6])) if len(sys.argv) > 6 else False

    f = ROOT.TFile.Open(input_file)
    if not f or f.IsZombie():
        print(f"Could not open ROOT file {input_file}")
        sys.exit(1)

    tr = f.Get(tree_name)
    if not tr:
        print(f"Could not find tree '{tree_name}' in file {input_file}")
        sys.exit(1)

    clusterPrint(tr, isEventsTree, max_workers=max_workers, validate=validate, per_branch=per_branch)
