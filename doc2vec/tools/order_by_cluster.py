import numpy as np
from typing import Tuple, Union
from .cluster_hdbscan import cluster_by_hdbscan
from queue import Queue


verbose = False

def _verbose_sims_each_testcase(sims, threshold_min, threshold_max, labels):
    print("thresh max = {}, min = {}".format(threshold_max, threshold_min))

    def _get_place(sim, t_min, t_max):
        if sim < t_min:
            return "tail"
        elif sim > t_max:
            return "head"
        else:
            return "middle"

    for i, sim in enumerate(sims):
        print(
            "testcase={}, sim={}, place={}, label={}".format(
                i, sim, _get_place(sim, threshold_min, threshold_max), labels[i]
            )
        )

""" Special queue for this order method. """
class MyQueue:
    def __init__(self, test_cases: list) -> None:
        self.size = len(test_cases)
        assert self.size, "Empty cluster(queue) checked."

        self.q = test_cases
        self.front = 0

    """ Return true if this queue is "empty" """
    def empty(self) -> bool:
        return self.front == self.size

    """ Sort testcases by descending similarity.(Inplace) """
    def sort(self) -> None:
        sorted_q = sorted(
            self.q,
            key=lambda pair: -pair[1]
        )
        self.q = sorted_q

    """ Overload __lt__ to support bulitin-func sorted. """
    def __lt__(self, other) -> bool:
        def _max_sim(q) -> float:
            return max([pair[1] for pair in q])

        return _max_sim(self.q) > _max_sim(other.q)

    """ Return a test_case(tuple)(pair) without delete it from queue. """
    def pop(self) -> tuple:
        assert not self.empty(), "Apply pop on a empty queue."

        test_case = self.q[self.front]
        self.front = self.front + 1
        return test_case

    def get_id(self) -> int:
        return self.q[0][2]

def _verbose_queue(queue):
    print("cluster label = {}".format(queue.get_id()))

    for t in queue.q:
        print("test case={}, sim={}, label={}".format(t[0], t[1], t[2]))
    print("")

""" Pair = [index, similarity, label] """
def _make_queues(pairs: Tuple[Tuple[int, float, int]], identifiers: list) -> list:
    global verbose

    # make queues
    queues = []

    for id_ in identifiers:

        cluster = [pair for pair in pairs if pair[2] == id_]
        queue = MyQueue(cluster)
        queue.sort()

        queues.append(queue)

        if verbose:
            _verbose_queue(queue)

    return sorted(queues)


def order_by_cluster(
        # parameters for order.
        sims: Union[list, np.ndarray],
        threshold_min: Union[float, int],
        threshold_max: Union[float, int],
        
        # parameters for cluster result.
        labels: Union[np.ndarray, list],
        identifiers: Union[np.ndarray, list],
        n_cluster: int
    ) -> list:

    global verbose

    # ensure np.ndarray
    sims, labels, identifiers = tuple(map(
        lambda x: np.asarray(x).ravel(),
        [sims, labels, identifiers]
    ))
    # sims = np.asarray(sims).ravel()
    # labels = np.asarray(labels).ravel()
    # identifiers = np.asarray(identifiers).ravel()

    # check dynamic threshold
    threshold_min = np.percentile(sims, threshold_min) \
        if threshold_min > 1 else threshold_min
    threshold_max = np.percentile(sims, threshold_max) \
        if threshold_max > 1 else threshold_max
    
    # print debug information.
    if verbose:
        _verbose_sims_each_testcase(sims, threshold_min, threshold_max, labels)

    # make pairs of each test case
    pairs = [(i, sims[i], labels[i]) for i in range(len(sims))]
    sorted_queues = _make_queues(pairs, identifiers)

    # order
    heads = []
    middles = []
    tails = []

    while True:
        # loop for each queue
        all_empty = True

        for queue in sorted_queues:
            if not queue.empty():
                all_empty = False
                pair = queue.pop()

                testcase = pair[0]
                sim = pair[1]

                if sim > threshold_max:
                    assert False
                    heads.append(pair) # append pair for the following sort.
                elif sim < threshold_min:
                    tails.append(pair)
                else:
                    middles.append(testcase)
        if all_empty:
            break

    # tails = sorted(tails, key=lambda x: -x[1])
    tails_ = [x[0] for x in tails]

    heads = sorted(heads, key=lambda x: -x[1])
    heads_ = [x[0] for x in heads]

    order = heads_ + middles + tails_

    return order, sorted_queues # clusters






