import pyarrow.parquet as pq, glob, os
D='/scratch/r919a03/code_verl_test/sources/rl_code_v1/data/parquet'
for f in sorted(glob.glob(D+'/*.parquet')):
    t=pq.read_table(f)
    print('=====',os.path.basename(f),'rows=',t.num_rows)
    print(t.schema)
    md=pq.read_metadata(f)
    print('  row_groups=',md.num_row_groups, 'created_by=',md.created_by)
