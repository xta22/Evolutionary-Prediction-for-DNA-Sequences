from pathlib import Path
import re

import pandas as pd
from Bio import Phylo


# ============================================================
# Files
# ============================================================

LSD2_FILE = Path(
    "dating_results/lsd2/Ara+2/Ara+2.date.nexus"
)

TT_TREE_FILE = Path(
    "dating_results/treetime/"
    "Ara+2_alignment_relaxed/timetree.nexus"
)

TT_DATES_FILE = Path(
    "dating_results/treetime/"
    "Ara+2_alignment_relaxed/dates.tsv"
)

OUT_LSD2 = Path(
    "Ara+2_LSD2_branches.csv"
)

OUT_TT = Path(
    "Ara+2_TreeTime_branches.csv"
)


# ============================================================
# TreeTime dates
# ============================================================

def load_treetime_dates():

    dates = {}

    with open(TT_DATES_FILE) as f:

        for line in f:

            line = line.strip()

            if (
                not line
                or line.startswith("#")
            ):
                continue

            parts = re.split(
                r"\s+",
                line
            )

            if len(parts) < 3:
                continue

            dates[parts[0]] = float(
                parts[2]
            )

    return dates


# ============================================================
# Get descendant tips
# ============================================================

def get_tips(clade):

    return tuple(
        sorted(
            tip.name
            for tip in clade.get_terminals()
        )
    )


# ============================================================
# TreeTime branch extraction
# ============================================================

def extract_treetime():

    tree = Phylo.read(
        TT_TREE_FILE,
        "nexus"
    )

    dates = load_treetime_dates()

    rows = []

    for parent in tree.find_clades(
        order="preorder"
    ):

        if parent.name is None:
            continue

        if parent.name not in dates:
            continue

        parent_date = dates[
            parent.name
        ]

        for child in parent.clades:

            if child.name is None:
                continue

            if child.name not in dates:
                continue

            child_date = dates[
                child.name
            ]

            rows.append({
                "parent":
                    parent.name,

                "child":
                    child.name,

                "parent_date":
                    parent_date,

                "child_date":
                    child_date,

                "delta_time":
                    child_date
                    - parent_date,

                "parent_tips":
                    "|".join(
                        get_tips(parent)
                    ),

                "child_tips":
                    "|".join(
                        get_tips(child)
                    ),

                "branch_length":
                    child.branch_length
            })

    df = pd.DataFrame(rows)

    df.to_csv(
        OUT_TT,
        index=False
    )

    return df


# ============================================================
# Extract date from BactDating/LSD2 comments
# ============================================================

def extract_date_from_comment(
    comment
):

    if comment is None:
        return None

    patterns = [
        r'date="([^"]+)"',
        r'date=([^,\]]+)',
    ]

    for pattern in patterns:

        m = re.search(
            pattern,
            comment
        )

        if m:
            try:
                return float(
                    m.group(1)
                )
            except ValueError:
                pass

    return None


# ============================================================
# LSD2 branch extraction
# ============================================================

def extract_lsd2():

    tree = Phylo.read(
        LSD2_FILE,
        "nexus"
    )

    rows = []

    for parent in tree.find_clades(
        order="preorder"
    ):

        parent_date = (
            extract_date_from_comment(
                getattr(
                    parent,
                    "comment",
                    None
                )
            )
        )

        if parent.is_terminal():

            # Terminal date should be encoded
            # in the comment.
            #
            # No outgoing branch.
            continue

        for child in parent.clades:

            child_date = (
                extract_date_from_comment(
                    getattr(
                        child,
                        "comment",
                        None
                    )
                )
            )

            if (
                parent_date is None
                or child_date is None
            ):
                continue

            rows.append({
                "parent":
                    parent.name,

                "child":
                    child.name,

                "parent_date":
                    parent_date,

                "child_date":
                    child_date,

                "delta_time":
                    child_date
                    - parent_date,

                "parent_tips":
                    "|".join(
                        get_tips(parent)
                    ),

                "child_tips":
                    "|".join(
                        get_tips(child)
                    ),

                "branch_length":
                    child.branch_length,

                "parent_comment":
                    getattr(
                        parent,
                        "comment",
                        None
                    ),

                "child_comment":
                    getattr(
                        child,
                        "comment",
                        None
                    )
            })

    df = pd.DataFrame(rows)

    df.to_csv(
        OUT_LSD2,
        index=False
    )

    return df


# ============================================================
# Run
# ============================================================

print("\n========================================")
print("ARA+2 DATED BRANCH EXTRACTION")
print("========================================")


tt = extract_treetime()

print(
    "\nTreeTime branches:",
    len(tt)
)

print(
    "TreeTime delta-time summary:"
)

if len(tt) > 0:
    print(
        tt["delta_time"].describe()
    )


lsd = extract_lsd2()

print(
    "\nLSD2 branches:",
    len(lsd)
)

print(
    "LSD2 delta-time summary:"
)

if len(lsd) > 0:
    print(
        lsd["delta_time"].describe()
    )


print(
    "\nSaved:"
)

print(
    OUT_TT
)

print(
    OUT_LSD2
)
