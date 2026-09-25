import sys
sys.path.insert(0, '.')
import phase7_evaluation as p7

m = p7.evaluate(report_path=None)
active = m['active']
for blk in active:
    for r in blk['records']:
        if 'MISSED_CONVERSATIONAL_MOVE' in r['_failures']:
            prev = r['prev_assistant'][:80] if r['prev_assistant'] else 'NONE'
            print(f"{blk['scenario'].name} T{r['turn']}: '{r['user']}' | mode={r['mode']} conf={r['confidence']} pause={r['pause']} ack={r['ack']} | gt_move={r['_anno'].move} strength={r['_anno'].strength} | prev='{r['prev_assistant'][:80]}'")