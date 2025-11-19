# darshan_reops
HEP-CCE amp;SOP darshan reops analysis

1. dump the cluster/branch basket

```python dump_cluster_boundary.py <paths-to-inputRootFile> <EventTreeName> <max_workers> <per_branch=0/1> <output=dump_{surfix}.json/dump_per_branch_{surfix}.json>```

2. map on to the offset in Darshan DXT records

```python <darshan_record> --include_names inputRootFile  --enable_mapping --tree_branch_file <dump_{surfix}.json/dump_per_branch_{surfix}.json>```

Note: The mapping checks if `per_branch_` is part of the `tree_branch_file` name to decide how to map the records and create the plots.
