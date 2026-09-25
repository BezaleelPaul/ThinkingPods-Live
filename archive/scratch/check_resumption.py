import sys
sys.path.insert(0, '.')
import phase7_evaluation as p7

m = p7.evaluate(report_path=None)
active = m['active']
for blk in active:
    for r in blk['records']:
        if 'FAILED_RESUMPTION' in r['_failures']:
            print(f"{blk['scenario'].name} T{r['turn']}: '{r['user'][:50]}'")
            print(f'  mode={r["mode"]} conf={r["confidence"]} pause={r["pause"]} ack={r["ack"]}')
            print(f'  obj_before={r["objective_before"]} obj={r["objective"]}')
            print(f'  prev_paused={r["_resumption_check"]}')
            print(f'  prev_obj={r["objective_before"]} cur_obj={r["objective"]}')