import itertools
from queue import PriorityQueue
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import distance
import time

from read import readInput
from write import writeOutput
from host import GO

class MyPlayer():

    def __init__(self):
        # self.type = 'ABPrune'
        self.prev_go = None 
        self.curr_go = None 
        self.piece_type = None 
        self.move_num = None
        self.size = None
        self.max_depth = None 
        self.start_time = None 

    def get_input(self, go, piece_type):
        '''
        Get one input.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input.
        '''        
        self.size = go.size
        self.curr_go = go.copy_board()
        self.piece_type = piece_type
        self.max_depth = 4
        self.move_num = self.get_move_num()

        self.start_time = time.time()
        move = self.alpha_beta(self.curr_go)
        end = time.time()
        print(f'Evaluation time: {round(end-self.start_time, 7)}s')

        # self.write_move(move)
        return move
    
    # driver
    def alpha_beta(self, go):

        valid_moves = self.possible_placements(go, self.piece_type)
        if valid_moves.empty(): return "PASS"
        
        curr_depth = 0
        heur_dict = {}
        maxv = -1000
        alpha = -1000
        beta = 1000

        while not valid_moves.empty(): 
            move = valid_moves.get()[1]

            val = self.driver_heur(go, move)
            ab = self.min_alpha_beta(self.curr_go, curr_depth+1, self.move_num+1, alpha, beta)                
            val+= ab

            alpha = max(val, alpha)

            heur_dict[val] = move
            
            maxv = max(maxv, val)

            self.curr_go = self.prev_go
        
        return heur_dict[maxv] if maxv > -1000 else "PASS"

    def max_alpha_beta(self, go, depth, move_num, alpha, beta): 

        valid_moves = self.possible_placements(go, self.piece_type) 
        maxv = -1000

        if self.is_terminal(depth, move_num, valid_moves): return self.heuristic(go, move_num) 

        while not valid_moves.empty(): 
            move = valid_moves.get()[1]
            # if move == "PASS" 

            self.curr_go, died = self.next_board_state(go, move[0], move[1], self.piece_type)
            died_weight = 10 if self.move_num >=21 else 7
            num_died = died_weight * len(died)
            
            bad_move = -self.is_bad_move(go, move[0], move[1], move_num, len(died))

            curr_eval = self.min_alpha_beta(self.curr_go, depth+1, move_num+1, alpha, beta) 
            
            curr_eval += num_died + bad_move

            maxv = max(maxv, curr_eval)
            if maxv >= beta: return maxv
            alpha = max(alpha, maxv)
            
            self.curr_go = self.prev_go

        return maxv

    def min_alpha_beta(self, go, depth, move_num, alpha, beta): 

        valid_moves = self.possible_placements(go, self.piece_type)
        minv = 1000
        player = 3 - self.piece_type

        if self.is_terminal(depth, move_num, valid_moves): return self.heuristic(go, move_num) 

        while not valid_moves.empty(): 
            move = valid_moves.get()[1]

            self.curr_go, died = self.next_board_state(go, move[0], move[1], player)
            died_weight = 10 if self.move_num >=21 else 7
            num_died = -died_weight * len(died)
            
            bad_move = self.is_bad_move(go, move[0], move[1], move_num, len(died))

            curr_eval = self.max_alpha_beta(self.curr_go, depth+1, move_num+1, alpha, beta)
            curr_eval += num_died + bad_move

            minv = min(minv, curr_eval)
            if minv <= alpha: return minv
            beta = min(beta, minv)

            self.curr_go = self.prev_go

        return minv

    # pq sorted by distance to center
    # hard codes first move to be center or directly 
    # top right diagnoal of center 
    def possible_placements(self, go, piece_type):
        center = int(np.floor(go.size / 2))
        center_index = (center, center)
        possible_placements = PriorityQueue()
        if self.move_num <= 2:
            if go.valid_place_check(center, center, piece_type):
                possible_placements.put((0, center_index))
            else: possible_placements.put((0, (center-1, center-1)))
            return possible_placements
        for i, j in itertools.product(range(go.size), range(go.size)):
            if go.valid_place_check(i, j, piece_type):
                dist_to_center = distance.euclidean(center_index, (i, j))
                possible_placements.put((dist_to_center, (i,j)))
        ## possible_placements: add Pass action with HIGH action 
        return possible_placements
    
    # returns board after move (i,j) is made
    # returns list of died pieces that result 
    def next_board_state(self, go, i, j, piece_type): 
        copy_go = go.copy_board()
        self.prev_go = copy_go 
        copy_go.place_chess(i, j, piece_type)
        died = copy_go.remove_died_pieces(3 - piece_type)
        return copy_go, died
    
    # gets move num for player from external file 
    # creates file if not exists 
    def get_move_num(self): 
        empty_board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        move_num = 1 if self.piece_type == 1 else 2
        
        if(self.curr_go.compare_board(self.curr_go.previous_board, empty_board)):
            with open("moveNum.txt", "w+") as fName:
                fName.write(str(move_num + 2))
                return move_num

        try:
            with open("moveNum.txt", 'r+') as fName:
                move_num = int(fName.read())
                fName.seek(0)
                fName.write(str(move_num + 2))
                fName.truncate()

        except (FileNotFoundError): 
            with open("moveNum.txt", 'w+') as fName:
                fName.write(str(move_num + 2))
        
        return move_num


    def heuristic(self, go, move_num):

        oppo_score = go.komi if self.piece_type == 1 else 0 
        self_score = go.komi if self.piece_type == 2 else 0 
        
        oppo_score += go.score(3-self.piece_type)
        self_score += go.score(self.piece_type)

        diff_score = self_score - oppo_score 
        if move_num >= 21:  # or move_action = PASS
            diff_score += 1000 if diff_score > 0 else -1000
        
        euler_weighted = -5 if move_num < 21 else -4
        euler_weighted *= self.euler(go)

        delta = 5
        libs_diff = self.count_all_libs(go)
        libs_diff = min(max(libs_diff, -delta), delta)

        # might want to weight avg dist to center 
        # return diff_score + euler_weighted + libs_diff + self.find_avg_dist_to_center(go, move_num)
        return diff_score + euler_weighted + libs_diff
    
    # finds libs for specified coord on board 
    def lib_locs(self, go, i, j):
        board = go.board
        neighbors =  set(go.detect_neighbor(i, j))
        
        libs = set()
        ally_members = set()
        oppos = set()

        for mem in neighbors:
            # if mem not in all_explored: 
            if board[mem[0]][mem[1]] == 0:
                libs.add(mem)
            elif board[mem[0]][mem[1]] == board[i][j]:
                ally_members.add(mem)
            else: ## oppo piece 
                oppos.add(mem)

        return ally_members, oppos, libs

    # calc diff in self and oppo 1st, 2nd, and 3rd order libs
    def count_all_libs(self, go): 
        self_first_libs = set()
        oppo_first_libs = set()
        my_pts = set() # self piece AND lib locs
        oppo_pts = set() # oppo piece AND lib locs
        board = go.board

        for i, j in itertools.product(range(go.size), range(go.size)):
            if board[i][j]!=0: 
                if board[i][j] == self.piece_type: 
                    ally_pieces, oppo_pieces, libs = self.lib_locs(go, i, j)
                    self_first_libs.update(libs)
                    my_pts.update(ally_pieces)
                    my_pts.update(libs)

                else: 
                    ally_pieces, oppo_pieces, libs = self.lib_locs(go, i, j)
                    oppo_first_libs.update(libs)
                    oppo_pts.update(ally_pieces)
                    oppo_pts.update(libs)
                
        self_sec_libs, self_num_third_libs = self.ordered_libs(go, self_first_libs)
        oppo_sec_libs, oppo_num_third_libs = self.ordered_libs(go, oppo_first_libs)

        sec_diff = len(self_sec_libs) - len(oppo_sec_libs)
        first_sec_diff = len(self_first_libs) - len(oppo_first_libs) + sec_diff

        third_diff = 0.5 * (self_num_third_libs - oppo_num_third_libs) 

        total = first_sec_diff + third_diff
        delta = 5

        total = min(max(total, -delta), delta)

        return  total
    
    # finds 2nd and 3rd order libs, doesnt count libs 
    # that appeared in prev lib sets
    def ordered_libs(self, go, first_libs): 
        second_libs = set()
        third_libs = set()
        all_libs = first_libs.copy()

        for lib in first_libs: 
            ally, oppo, sec_libs = self.lib_locs(go, lib[0], lib[1])
            for sec_lib in sec_libs: 
                if sec_lib not in first_libs: 
                    second_libs.add(sec_lib)

        all_libs.update(second_libs)
        
        for lib in second_libs:
            ally, oppo, th_libs = self.lib_locs(go, lib[0], lib[1])
            for th_lib in th_libs: 
                if th_lib not in all_libs: 
                    third_libs.add(th_lib)
        
        return second_libs, len(third_libs)        
    
    # avoid playing on edges/corners with no nearby allies
    # avoid immediate recapture if num_died=1
    def is_bad_move(self, go, i, j, move_num, num_died):

        if move_num <= 2 or num_died > 1: return 0

        allys, oppos, my_free = self.lib_locs(go, i, j)

        # avoid play on edges/corners if no neighbor allies
        # avoid play where 1 lib and more than one neighbor oppo if no neighbor ally
        if len(allys) == 0: 
            if (i in [0, 4] or j in [0, 4]): 
                return 5 if move_num >=21 else 15
            if len(my_free) == 1 and len(oppos) >= 1:
                return 15
        
        # immediate re-capture possible
        if len(my_free)==1 and num_died == 1: 
            return 15

        return -10

    # generate binary board for self and oppo 
    # used in euler 
    def binary_board(self, go): 
        board = go.board
        size = go.size
        self_binary = np.zeros((size, size), dtype=int)
        oppo_binary = np.zeros((size, size), dtype=int)

        for i, j in itertools.product(range(size), range(size)):
            if board[i][j]==self.piece_type: 
                self_binary[i,j]=1
            elif board[i][j]==3-self.piece_type:
                oppo_binary[i,j]=1
        return self_binary, oppo_binary
    

    def euler(self, go):
        self_binary, oppo_binary = self.binary_board(go)

        coefs = [0, 1, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0]

        config = np.array([[0, 0, 0], [0, 1, 4], [0, 2, 8]])

        self_conv = ndi.convolve(self_binary, config, mode='constant', cval=0)
        oppo_conv = ndi.convolve(oppo_binary, config, mode='constant', cval=0)
        
        self_count = np.bincount(self_conv.ravel(), minlength=16)
        oppo_count = np.bincount(oppo_conv.ravel(), minlength=16)

        return (coefs @ self_count) - (coefs @ oppo_count)

    # modified max heuristic for driver 
    # weights num_died pieces, current diff in libs, 
    # if bad move, and euler num on board
    def driver_heur(self, go, move): 
        
        self.curr_go, died = self.next_board_state(go, move[0], move[1], self.piece_type)
        died_weight = 10 if self.move_num >=21 else 7
        num_died = died_weight * len(died)

        bad_move = -self.is_bad_move(go, move[0], move[1], self.move_num, len(died))

        delta = 5
        libs_diff = self.count_all_libs(self.curr_go)
        libs_diff = min(max(libs_diff, -delta), delta)

        euler_weight = -5 if self.move_num < 21 else -4
        euler = euler_weight * self.euler(self.curr_go)

        # might want to weight avg dist to center 
        # return bad_move + libs_diff + num_died + euler + self.find_avg_dist_to_center(go, self.move_num)
        return bad_move + libs_diff + num_died + euler 
        
    # tests if max depth has been reached, timer has run out, 
    # max move_num has been reached, or no valid moves 
    def is_terminal(self, depth, move_num, valid_moves):
        curr_time = time.time() - self.start_time 
        if self.max_depth == depth or move_num > go.max_move or valid_moves.empty() or curr_time > 8:
            return True

        return False 

    # finds self and oppo avg location on the board
    # returns the diff of dist to center from avg locs
    def find_avg_dist_to_center(self, go, move_num):     

        if self.piece_type == 2 or move_num == 1: return 0

        board = go.board
        center = int(np.floor(go.size / 2))
        center_index = (center, center)

        self_sum_loc= [0,0]
        self_num_pieces = 0

        oppo_sum_loc = [0,0]
        oppo_num_pieces = 0

        for i, j in itertools.product(range(go.size), range(go.size)):
            if board[i][j] == self.piece_type: 
                self_sum_loc[0] += i 
                self_sum_loc[1] += j
                self_num_pieces += 1
            
            elif board[i][j] == 3 - self.piece_type:
                oppo_sum_loc[0] += i 
                oppo_sum_loc[1] += j
                oppo_num_pieces += 1
        
        self_avg_x = self_sum_loc[0] / self_num_pieces
        self_avg_y = self_sum_loc[1] / self_num_pieces

        oppo_avg_x = oppo_sum_loc[0] / oppo_num_pieces
        oppo_avg_y = oppo_sum_loc[1] / oppo_num_pieces

        return distance.euclidean(center_index, (self_avg_x, self_avg_y)) - distance.euclidean(center_index, (oppo_avg_x, oppo_avg_y))


if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    player = MyPlayer()
    action = player.get_input(go, piece_type)
    writeOutput(action)


        # def write_move(self, move):
        # empty_board = [[0 for _ in range(self.size)] for _ in range(self.size)]

        # try:
        #     with open("movesMade.txt", 'a') as fName: 
        #         if(self.curr_go.compare_board(self.curr_go.previous_board, empty_board)):
        #             lines = ["new round: \n", str(move), "\n"]
        #             fName.writelines(lines)

        #         else: fName.write(str(move) + "\n")

        # except (FileNotFoundError): 
        #     lines = ["new round: \n", str(move), "\n"]

        #     with open("movesMade.txt", 'w+') as fName:
        #         fName.writelines(lines)