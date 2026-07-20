"""consent_text.py — Versioned acknowledgment text shown before the survey.

The researcher must replace [RESEARCHER_NAME] with their actual name before launch.
verify_setup.py will refuse to pass if that placeholder is still present.
"""

ACK_V1 = """
**Quick acknowledgment before you start**

This is a 30-question study for an IB Extended Essay on decision-making.
It records only your choices and how long you took to answer — no name, no
email, nothing else. It takes about 15 minutes.

You're free to stop at any time by closing the tab. After you finish you'll
see a unique ID; message [RESEARCHER_NAME] with that ID if you ever want
your data removed.
"""

VERSION = "v1"
