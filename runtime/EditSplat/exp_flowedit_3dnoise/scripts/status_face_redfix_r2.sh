#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import re,glob,os
logs=sorted(glob.glob('/dev-vepfs/rc_wu/rc_wu/edit/EditSplat/exp_flowedit_3dnoise/logs/full_redfix_face_*_20260303_081917.log'))
for p in logs:
    txt=open(p,'r',encoding='utf-8',errors='ignore').read()
    m_pair=list(re.finditer(r'Multi-view reprojection progress:\s*\d+%\|[^\n]*?\|\s*(\d+)/56',txt))
    m_edit=list(re.finditer(r'Initial editing progress:\s*(\d+)%\|',txt))
    done=int(m_pair[-1].group(1)) if m_pair else 0
    ep=int(m_edit[-1].group(1)) if m_edit else 0
    print(f"{os.path.basename(p)}\tedit={ep}%\treproj={done}/56")
PY

n=$(pgrep -f "run_editing_flow_(baseline|3dnoise)_wrapper.py .*full_real_face_" | wc -l || true)
echo "running_jobs=$n"
