import dualPriorityQueue as dpq
import collections

EDGE_WEIGHT = 1
Unchanged = None


class DStarLite:

    def __init__(self, problem, heuristic_func):
        self._h = heuristic_func
        self._U = dpq.DualPriorityQueue()
        self._km = 0
        self._start = problem.getStartState()
        self._last = problem.getStartState()
        self._goal = problem.getGoalState()
        self._path = []
        self._changed_edges = list()
        self._pop_count = 0

        self._has_path = False
        self._best_path = None
        self._last_path = None

        # set up the map/grid
        x_res, y_res = problem.getDims()
        self._width = x_res
        self._height = y_res
        self._node_count = x_res * y_res
        self._vertex_costs = [[[float("inf"), float("inf")] for _ in range(y_res)] for _ in range(x_res)]
        self._is_wall = problem.getNaiveWalls()

        # init the goal node (works backward)
        self._set_weight_tuple(self._goal, (Unchanged, 0))
        prim, sec = self.compute_keys(self._goal)
        self._U.push(key=self._goal, primary=prim, secondary=sec)

        self.compute_shortest_path()


    def update_vertex(self, coord, exclusion_node=None):
        if exclusion_node is None:
            exclusion_node = self._start

        if coord != exclusion_node:
            new_rhs = float("inf")
            if not self._is_wall[coord[0]][coord[1]]:
                neighbors = self._get_neighbors(coord)
                for each in neighbors:
                    g, _ = self._get_weight_tuple(each)
                    new_rhs = min(new_rhs, g + EDGE_WEIGHT)
            self._set_weight_tuple(coord, (Unchanged, new_rhs))

        self._U.delete_key(coord)

        g, rhs = self._get_weight_tuple(coord)
        if g != rhs:
            prim, sec = self.compute_keys(coord)
            self._U.push(key=coord, primary=prim, secondary=sec)


    def compute_keys(self, coord):
        h_cost = self._h(self._start, coord)
        cost_tuple = self._get_weight_tuple(coord)
        secondary = min(cost_tuple)
        primary = secondary + h_cost + self._km
        return primary, secondary


    def _get_neighbors(self, coord):
        all_dirs = [(coord[0], coord[1] + 1),
                    (coord[0] + 1, coord[1]),
                    (coord[0], coord[1] - 1),
                    (coord[0] - 1, coord[1])]
        valid_dirs = []
        for each in all_dirs:
            if self._in_map(each):
                valid_dirs.append(each)
        return valid_dirs


    def compute_shortest_path(self):
        g_start, rhs_start = self._get_weight_tuple(self._start)
        while (self._U.size() > 0 and self._tuple_lt(self._U.peek(), self.compute_keys(self._start))) or \
                g_start != rhs_start:
            k_old = self._U.peek()
            u = self._U.pop()[0]
            self._pop_count += 1
            g_u, rhs_u = self._get_weight_tuple(u)
            if self._tuple_lt(k_old, self.compute_keys(u)):
                prim, sec = self.compute_keys(u)
                self._U.push(u, prim, sec)
            elif g_u > rhs_u:
                self._set_weight_tuple(u, (rhs_u, Unchanged))
                for each in self._get_neighbors(u):
                    self.update_vertex(each, self._goal)
            else:
                self._set_weight_tuple(u, (float("inf"), Unchanged))
                for each in self._get_neighbors(u):
                    self.update_vertex(each, self._goal)
                self.update_vertex(u, self._goal)

            g_start, rhs_start = self._get_weight_tuple(self._start)
        self._has_path = True


    def make_wall_at(self, coord):
        x, y = coord

        start_neighbors = self._get_neighbors(self._start)
        if coord not in start_neighbors:
            raise ValueError("A wall cannot be discovered at a non-adjacent location; this breaks D* Lite.")
        self._changed_edges.append(coord)

        self._is_wall[x][y] = True
        for each in self._get_neighbors(coord):
            self._changed_edges.append(each)

        self._km += self._h(self._last, self._start)
        self._last = self._start
        for each in self._changed_edges:
            self.update_vertex(each, self._goal)
        self._changed_edges = list()
        self.compute_shortest_path()


    def take_step(self):
        if self._start == self._goal:
            return self._start

        g_start, rhs_start = self._get_weight_tuple(self._start)
        if g_start == float("inf"):
            return self._start

        argmin = (None, float("inf"))
        for each in self._get_neighbors(self._start):
            weight = EDGE_WEIGHT + self._get_weight_tuple(each)[0]
            if weight < argmin[1]:
                argmin = (each, weight)
        self._path.append(self._start)
        self._start = argmin[0]

        return self._start


    def extract_path(self, placeholder=None):
        return super(DStarLite, self).extract_path(backward=False)


    def get_route(self):
        res = list(self._path)
        res.append(self._start)
        return res


    def _set_weight_tuple(self, coord, tup):
        x, y = coord
        if tup[0] is not Unchanged:
            self._vertex_costs[x][y][0] = tup[0]
        if tup[1] is not Unchanged:
            self._vertex_costs[x][y][1] = tup[1]


    def _get_weight_tuple(self, coord):
        x, y = coord
        return self._vertex_costs[x][y]


    def _tuple_lt(self, tup1, tup2):
        if not isinstance(tup1, collections.Iterable):
            raise ValueError("Left-side tuple is not iterable: {0}".format(tup1))

        if not isinstance(tup2, collections.Iterable):
            raise ValueError("Right-side tuple is not iterable: {0}".format(tup2))

        if len(tup1) == 2:
            t1_primary, t1_secondary = tup1
        elif len(tup1) == 3:
            t1_label, t1_primary, t1_secondary = tup1
        else:
            raise ValueError("Left-side tuple contains unexpected arity: {0}".format(tup1))

        if len(tup2) == 2:
            t2_primary, t2_secondary = tup2
        elif len(tup2) == 3:
            t2_label, t2_primary, t2_secondary = tup2
        else:
            raise ValueError("Right-side tuple contains unexpected arity: {0}".format(tup2))

        if t1_primary < t2_primary:
            return True
        elif t1_primary > t2_primary:
            return False
        else:
            return t1_secondary < t2_secondary


    def _in_map(self, coord):
        x, y = coord
        return 0 <= x < self._width and 0 <= y < self._height


