import qelos_core as q
from qelos_core.scripts.query.wikisql_clean import get_accuracies, get_avg_accs_of


def run(p="none"):
    assert(p != "none")
    get_accuracies(p)


if __name__ == "__main__":
    q.embed()
    # q.argprun(run)