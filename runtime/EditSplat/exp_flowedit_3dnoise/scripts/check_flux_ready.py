#!/usr/bin/env python3
from pathlib import Path

cache = Path('/dev-vepfs/rc_wu/rc_wu/cache/hf_home/hub/models--black-forest-labs--FLUX.1-dev/blobs')
need = {
    'ec87bffd1923e8b2774a6d240c922a41f6143081d52cf83b8fe39e9d838c893e',
    'a5640855b301fcdbceddfa90ae8066cd9414aff020552a201a255ecf2059da00',
    'd86a3038eacaa720682cb9b1da3c49fecf8a3ded605af4def6061eaa18903eb8',
    '5e830704a83aa938dfaf23da308100a1c44b83fa084283abf1d163ea727e5f7a',
    '0d9c7c663217d1c3d44a6deed4e1cf1ac09fbc2c4137c47de1e3d74c959833de',
}
missing = []
incomplete = []
for h in sorted(need):
    if not (cache / h).exists():
        if (cache / f"{h}.incomplete").exists():
            incomplete.append(h)
        else:
            missing.append(h)

print('missing', missing)
print('incomplete', incomplete)
print('ready', (len(missing)==0 and len(incomplete)==0))
