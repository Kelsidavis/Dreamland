"""Legacy import shim: `towel` was renamed to `dreamland`.

User-authored skills and scripts written as `from towel.skills.base
import Skill` keep working — this package aliases itself to the
dreamland package, and Python's import machinery resolves submodules
(`towel.skills.base` → `dreamland/skills/base.py`) through the aliased
module's __path__.
"""

import sys

import dreamland

sys.modules[__name__] = dreamland
