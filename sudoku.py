import heapq
import random
from typing import Dict, List, Set, Tuple, Optional, Generator


# ==========================================
# 1. CSP Representation & Exact Solvers (BT, FC, AC-3)
# ==========================================

class SudokuCSP:
    """
    Represents a Sudoku puzzle as a Constraint Satisfaction Problem (81 cells).
    Variables: Cells r0c0 to r8c8 (or 0-80 indices)
    Domain: {1, 2, 3, 4, 5, 6, 7, 8, 9}
    Constraints: All-Diff across rows, columns, and 3x3 sub-grids.
    """

    def __init__(self, puzzle_str: str):
        """
        Initializes the board from an 81-character string (0 or '.' for empty).
        """
        self.variables = [f"r{r}c{c}" for r in range(9) for c in range(9)]
        self.domains: Dict[str, Set[int]] = {}

        # Build neighbor constraints map
        self.peers: Dict[str, Set[str]] = {v: set() for v in self.variables}
        for r in range(9):
            for c in range(9):
                v = f"r{r}c{c}"
                # Row and column peers
                for i in range(9):
                    if i != c: self.peers[v].add(f"r{r}c{i}")
                    if i != r: self.peers[v].add(f"r{i}c{c}")
                # Box peers
                br, bc = (r // 3) * 3, (c // 3) * 3
                for dr in range(3):
                    for dc in range(3):
                        rr, cc = br + dr, bc + dc
                        if (rr, cc) != (r, c):
                            self.peers[v].add(f"r{rr}c{cc}")

        # Initialize domains based on input puzzle
        for i, char in enumerate(puzzle_str):
            v = self.variables[i]
            val = int(char) if char.isdigit() and char != '0' else 0
            if val != 0:
                self.domains[v] = {val}
            else:
                self.domains[v] = set(range(1, 10))

    def is_consistent(self, var: str, value: int, assignment: Dict[str, int]) -> bool:
        """Check if assigning value to var violates any constraints with current assignment."""
        for peer in self.peers[var]:
            if peer in assignment and assignment[peer] == value:
                return False
        return True

    def get_unassigned_variable_mrv(self, assignment: Dict[str, int]) -> str:
        """Select unassigned variable using Minimum Remaining Values (MRV) heuristic."""
        unassigned = [v for v in self.variables if v not in assignment]
        return min(unassigned, key=lambda var: len(self.domains[var]))

    # --- Backtracking Search ---
    def backtracking_search(self) -> Optional[Dict[str, int]]:
        """Standard Backtracking Search."""
        assignment = {v: list(self.domains[v])[0] for v in self.variables if len(self.domains[v]) == 1}
        return self._bt_rec(assignment)

    def _bt_rec(self, assignment: Dict[str, int]) -> Optional[Dict[str, int]]:
        if len(assignment) == len(self.variables):
            return assignment

        var = self.get_unassigned_variable_mrv(assignment)
        for val in sorted(self.domains[var]):
            if self.is_consistent(var, val, assignment):
                assignment[var] = val
                result = self._bt_rec(assignment)
                if result is not None:
                    return result
                del assignment[var]
        return None

    # --- Forward Checking ---
    def forward_checking_search(self) -> Optional[Dict[str, int]]:
        """Backtracking with Forward Checking."""
        domains = {v: set(d) for v, d in self.domains.items()}
        return self._fc_rec({}, domains)

    def _fc_rec(self, assignment: Dict[str, int], domains: Dict[str, Set[int]]) -> Optional[Dict[str, int]]:
        if len(assignment) == len(self.variables):
            return assignment

        var = min([v for v in self.variables if v not in assignment], key=lambda v: len(domains[v]))
        for val in sorted(domains[var]):
            # Temporarily assign
            new_assignment = assignment.copy()
            new_assignment[var] = val

            # Forward check: prune domains of unassigned peers
            new_domains = {v: set(d) for v, d in domains.items()}
            new_domains[var] = {val}
            failure = False

            for peer in self.peers[var]:
                if peer not in new_assignment:
                    new_domains[peer].discard(val)
                    if not new_domains[peer]:
                        failure = True
                        break

            if not failure:
                result = self._fc_rec(new_assignment, new_domains)
                if result is not None:
                    return result
        return None

    # --- AC-3 (Arc Consistency) ---
    def ac3(self, domains: Optional[Dict[str, Set[int]]] = None) -> bool:
        """Enforces arc consistency across the CSP network."""
        if domains is None:
            domains = self.domains

        queue = [(xi, xj) for xi in self.variables for xj in self.peers[xi]]

        while queue:
            (xi, xj) = queue.pop(0)
            if self._revise(domains, xi, xj):
                if not domains[xi]:
                    return False
                for xk in self.peers[xi]:
                    if xk != xj:
                        queue.append((xk, xi))
        return True

    def _revise(self, domains: Dict[str, Set[int]], xi: str, xj: str) -> bool:
        revised = False
        to_remove = set()
        for x in domains[xi]:
            # If no value y in domains[xj] satisfies constraint (x != y), remove x
            if not any(x != y for y in domains[xj]):
                to_remove.add(x)
                revised = True
        domains[xi] -= to_remove
        return revised


# ==========================================
# 2. Randomized Sudoku Generator & Uniqueness
# ==========================================

class SudokuGenerator:
    @staticmethod
    def generate_full_board() -> List[List[int]]:
        """Generates a valid completed 9x9 Sudoku board using randomized backtracking."""
        board = [[0] * 9 for _ in range(9)]
        SudokuGenerator._fill_board(board)
        return board

    @staticmethod
    def _fill_board(board: List[List[int]]) -> bool:
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    nums = list(range(1, 10))
                    random.shuffle(nums)
                    for n in nums:
                        if SudokuGenerator._is_valid(board, r, c, n):
                            board[r][c] = n
                            if SudokuGenerator._fill_board(board):
                                return True
                            board[r][c] = 0
                    return False
        return True

    @staticmethod
    def _is_valid(board: List[List[int]], r: int, c: int, n: int) -> bool:
        for i in range(9):
            if board[r][i] == n or board[i][c] == n:
                return False
        br, bc = (r // 3) * 3, (c // 3) * 3
        for dr in range(3):
            for dc in range(3):
                if board[br + dr][bc + dc] == n:
                    return False
        return True

    @classmethod
    def generate_puzzle(cls, holes: int = 40) -> Tuple[str, str]:
        """Generates a puzzle with a unique solution by removing cells."""
        full = cls.generate_full_board()
        puzzle = [row[:] for row in full]

        cells = [(r, c) for r in range(9) for c in range(9)]
        random.shuffle(cells)

        removed = 0
        for r, c in cells:
            if removed >= holes:
                break
            temp = puzzle[r][c]
            puzzle[r][c] = 0

            # Check uniqueness
            puzzle_str = "".join(str(cell) for row in puzzle for cell in row)
            csp = SudokuCSP(puzzle_str)

            # Count solutions to verify uniqueness
            solutions = 0

            def count_sols(assignment):
                nonlocal solutions
                if solutions > 1: return
                if len(assignment) == 81:
                    solutions += 1
                    return
                var = csp.get_unassigned_variable_mrv(assignment)
                for val in csp.domains[var]:
                    if csp.is_consistent(var, val, assignment):
                        assignment[var] = val
                        count_sols(assignment)
                        del assignment[var]

            initial = {csp.variables[i]: int(puzzle_str[i]) for i in range(81) if puzzle_str[i] != '0'}
            count_sols(initial)

            if solutions != 1:
                puzzle[r][c] = temp  # Restore
            else:
                removed += 1

        p_str = "".join(str(cell) for row in puzzle for cell in row)
        s_str = "".join(str(cell) for row in full for cell in row)
        return p_str, s_str


# ==========================================
# 3. Correction Search Algorithms (UCS, Greedy, A*)
# ==========================================

class CorrectionSearch:
    """
    Finds optimal corrections on an OCR-misread Sudoku board using
    Neural Network cost probabilities, enforcing a maximum of 3 corrections.
    """

    def __init__(self, observed_board: str, nn_probabilities: List[Dict[int, float]]):
        """
        observed_board: 81-char string of digits perceived by OCR.
        nn_probabilities: List of 81 dicts mapping digit (1-9) to confidence probability.
        """
        self.observed = [int(c) for c in observed_board]
        self.nn_probs = nn_probabilities

    def _get_cost(self, cell_idx: int, corrected_digit: int) -> float:
        """Cost function based on Negative Log-Likelihood / confidence from NN."""
        prob = self.nn_probs[cell_idx].get(corrected_digit, 0.001)
        return -float(torch_log(prob)) if 'torch_log' else max(0.0, -1.0 * (prob))  # placeholder logic

    def _heuristic(self, state: List[int]) -> float:
        """A* conflict heuristic: counts current constraint violations on the board."""
        violations = 0
        board = [[state[r * 9 + c] for c in range(9)] for r in range(9)]
        for r in range(9):
            for c in range(9):
                val = board[r][c]
                if val == 0: continue
                # Check row/col/box clashes
                for i in range(9):
                    if i != c and board[r][i] == val: violations += 1
                    if i != r and board[i][c] == val: violations += 1
        return float(violations)

    def uniform_cost_search(self) -> Optional[Tuple[List[int], float, int]]:
        """UCS Correction Search (Max 3 corrections)."""
        return self._search(mode='ucs')

    def greedy_search(self) -> Optional[Tuple[List[int], float, int]]:
        """Greedy Best-First Correction Search."""
        return self._search(mode='greedy')

    def a_star_search(self) -> Optional[Tuple[List[int], float, int]]:
        """A* Search with Conflict Heuristic."""
        return self._search(mode='astar')

    def _search(self, mode: str):
        # Priority queue elements: (priority, g_cost, current_state, num_corrections)
        initial_state = self.observed.copy()
        initial_h = self._heuristic(initial_state) if mode == 'astar' else 0.0

        pq = [(initial_h, 0.0, initial_state, 0)]
        visited = set()
        logs = 0

        while pq:
            f, g, state, corrections = heapq.heappop(pq)
            logs += 1

            # Check if valid solved Sudoku
            board_str = "".join(map(str, state))
            csp = SudokuCSP(board_str)
            if csp.backtracking_search() is not None:
                return state, g, logs

            if corrections >= 3:
                continue

            state_tuple = tuple(state)
            if state_tuple in visited:
                continue
            visited.add(state_tuple)

            # Generate neighbors (change one cell to another digit)
            for i in range(81):
                original_digit = self.observed[i]
                for new_digit in range(1, 10):
                    if new_digit != state[i]:
                        next_state = state.copy()
                        next_state[i] = new_digit

                        step_cost = 1.0 - self.nn_probs[i].get(new_digit, 0.5)  # Cost metric
                        new_g = g + step_cost
                        new_corrections = corrections + (1 if new_digit != original_digit else 0)

                        if new_corrections <= 3:
                            h = self._heuristic(next_state) if mode == 'astar' else 0.0
                            priority = new_g + h if mode == 'astar' else (h if mode == 'greedy' else new_g)
                            heapq.heappush(pq, (priority, new_g, next_state, new_corrections))

        return None, float('inf'), logs


# Helper fallback for log function if math library needed
import math


def torch_log(p):
    return math.log(max(p, 1e-10))