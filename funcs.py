# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 23:27:23 2022

@author: gianl
"""

    
### TODO:
    # increase depth 
        # - check how it behaves with ENGINE LIMIT up.
        # - see if we maybe want to control it directly
    # improve leniency criterion
    ### -> idea for these two: find a range of moves from absolute best engine at best depth
    # improve book moves -> book moves are not convex; instead, they sometimes have holes!
    # analyse at scale, comparing with other players.
    # use older machines (stockfish 7 up to 14, for instance.)    

#### VARIOUS IMPORTS #####
import time
import pandas as pd

import chess
import chess.pgn
import chess.engine
import chess.polyglot
    
######### FILE STRUCTURE #######
import os
mainPath = r'C:\Users\gianl\Documents\GitHub\ChessEvaluations'
dataPath = os.path.join(mainPath, 'data')
enginePath = os.path.join(mainPath, 'engine', 'executables')
openingPath = os.path.join(mainPath, 'openings')
savePath_raw = os.path.join(mainPath, 'outputs_raw')
savePath_agg = os.path.join(mainPath, 'outputs_aggregated')

####### FIND GAMES IN PGN OBJECT #########
def get_offsets(pgnStr, filterDict):
    
    with open(os.path.join(dataPath, pgnStr)) as pgn:
        
        good_offsets = []
        
        while True:
            offset = pgn.tell()
    
            headers = chess.pgn.read_headers(pgn)
            if headers is None:
                break
    
            match = True
            for field, search_word in filterDict.items():
                if not search_word in headers.get(field, '?'):
                    match = False
            
            if match:
                good_offsets.append(offset)
        
    return good_offsets

def read_offsets(pgnStr, offsets):
    
    games = []
    with open(os.path.join(dataPath, pgnStr)) as pgn:
        for offset in offsets:
            pgn.seek(offset)
            game = chess.pgn.read_game(pgn)
            games.append(game)
        return games   


######### HELPERS ###########
def get_exp(score, model = 'sf', ply = 30):
    wdl = score.wdl(model = model, ply = ply)
    w, d, l = wdl
    exp = (w + 0.5 * d) / (w + d + l)
    return exp

def add_exp(infoDicts, model = 'sf', ply = 30):
    return [{**infoDict, 'exp': get_exp(infoDict['score'], model = model, ply = ply)} for infoDict in infoDicts]

def discard_inaccurate(infoDicts, 
                       leniency,
                       # exp_leniency = 0.05, cpl_leniency = 100, ### expected outcome, centipawn loss.
                       verbose = False):
    outDicts = []    
    
    best_move_exp = infoDicts[0]['exp']

    if leniency > 0:
        for infoDict in infoDicts:
            if best_move_exp - infoDict['exp'] < 0:
                raise Exception ### unordered outcomes!!!
            if best_move_exp - infoDict['exp'] <= leniency:
                outDicts.append(infoDict)
    else: ### take engine move
        outDicts = [infoDicts[0]]
  
    if verbose: print(f'Candidate variations: {len(infoDicts)}, of which acceptable: {len(outDicts)}')
    
    return outDicts

def right_adjust_str_to_length(myStr, target_length):
    curr_length = 1
    
    while curr_length < target_length:
        if len(myStr) == curr_length:
            myStr = ' ' + myStr
        curr_length += 1
    
    return myStr

####### FUNCTIONS THAT RUN ENGINES ON A GAME ########
def get_white_black_accuracy(game, book, engine,
                        engine_limit = 1, engine_lines = 5, 
                        leniency = 0.01):
    """
    Parameters
    ----------
    game : chess.pgn.Game
        A game of chess.
    book : chess.polyglot.MemoryMappedReader
        An opening book
    engine : chess.engine.SimpleEngine
        A chess engine instance
    engine_limit : float
        The time limit on each move in seconds. The default is 1.
    engine_lines : int, optional
        The number of best engine lines, sorted, to output. The default is 5.
    leniency : float, optional
        The leniency in terms of expected game outcome, where W = 1, D = 0.5, L = 0. 
        If a move is less than {leniency} worse than the best move, it is accepted.
        
    Returns
    -------

    white_accuracy : float
        White Player Engine Correlation % 
    black_accuracy : TYPE
        Black Player Engine Correlation % 
    """
    
    
    ### understand input
    # usually, we'd enter an engine instance, but we might want to initialize the engine inside here
    if isinstance(engine, str):
        engine = chess.engine.SimpleEngine.popen_uci(os.path.join(enginePath, engine))
    
    board = chess.Board()
    
    non_book_counters = {'white': 0, 'black': 0}
    engine_counters = {'white': 0, 'black': 0}
        
    for move in game.mainline_moves():
        
        san = board.san(move)
        color = 'white' if board.turn else 'black'
        ply = board.ply()
        plyStr = f'{int(ply / 2)}.' if ply % 2 == 0 else '  '
        
        plyStr = right_adjust_str_to_length(plyStr, 4)
        sanStr = right_adjust_str_to_length(san, 5)
        
        
        book_entries = book.find_all(board)
        book_moves = [entry.move for entry in book_entries]
        
        if move in book_moves:
            print(f'{plyStr} {sanStr}: Book Move!')
        else:
            infoDicts = engine.analyse(board, chess.engine.Limit(time = engine_limit), 
                                        multipv = engine_lines)
            ### add the expectation to the infoDict
            infoDicts = add_exp(infoDicts, model = 'sf', ply = 30)
    
            ### keep and filter out good moves
            goodDicts = discard_inaccurate(infoDicts, leniency = leniency)      
            goodMoves = [infoDict['pv'][0] for infoDict in goodDicts]
            
            ### get depth of primary variation
            depth = goodDicts[0]['depth']
            
            ### increment move quality counters
            non_book_counters[color] += 1
            
            if move in goodMoves:
                engine_counters[color] += 1
                print(f'{plyStr} {sanStr}: Engine Move at depth {depth}!')
            else:
                print(f'{plyStr} {sanStr}: Suboptimal move at depth {depth}!')
                
        board.push(move)
    
    white_accuracy = engine_counters['white'] / non_book_counters['white']
    black_accuracy = engine_counters['black'] / non_book_counters['black']
    
    print(f'White Accuracy: {white_accuracy}')
    print(f'Black Accuracy: {black_accuracy}')

    return game.headers, white_accuracy, black_accuracy


def get_move_evals(game, book, engine,
                   engine_limit = 1, engine_lines = 5):
    
    """
    Parameters
    ----------
    game : chess.pgn.Game
        A game of chess.
    book : chess.polyglot.MemoryMappedReader
        An opening book
    engine : chess.engine.SimpleEngine
        A chess engine instance
    engine_limit : float
        The time limit on each move in seconds. The default is 1.
    engine_lines : int, optional
        The number of best engine lines, sorted, to output. The default is 5.
    
    Returns
    -------

    outFrame: A comprehensive, move-by-move table outlining the data specified
    in moveDict (see below); i.e. move, eval, best move eval, move rank, etc.
    """
    
    
    start = time.time()
    
    ### understand input
    # usually, we'd enter an engine instance, but we might want to initialize the engine inside here
    if isinstance(engine, str):
        engine = chess.engine.SimpleEngine.popen_uci(os.path.join(enginePath, engine))
    
    board = chess.Board()
    
    outSeries = []
    
    ### get header info
    white_player = game.headers['White']
    black_player = game.headers['Black']
    white_elo = game.headers['WhiteElo']
    black_elo = game.headers['BlackElo']
    
    event = game.headers['Event']
    date = game.headers['Date']
    game_round = game.headers['Round']
    game_result = game.headers['Result']
    
    print(f'Analysing {white_player} against {black_player} at {event} ...')
    
    for move in game.mainline_moves():
        
        san = board.san(move)
        color = 'white' if board.turn else 'black'
        ply = board.ply()
        
        book_entries = book.find_all(board)
        book_moves = [entry.move for entry in book_entries]
        
        player = white_player if color == 'white' else black_player
        elo = white_elo if color == 'white' else black_elo
        
        if move in book_moves:
            rank = 'B'
            engine_eval = '_'
            primary_engine_eval = '_'
            best_san = '_'
            depth_min = '_'
            depth_max = '_'
            depth_move = '_'
            depth_best = '_'
        
        else:
            infoDicts = engine.analyse(board, chess.engine.Limit(time = engine_limit), 
                                        multipv = engine_lines)
            ### add the expectation to the infoDict
            infoDicts = add_exp(infoDicts, model = 'sf', ply = 30)
            moves = [infoDict['pv'][0] for infoDict in infoDicts]
    
            ### get depths
            depths = [infoDict['depth'] for infoDict in infoDicts]
            depth_min, depth_max = min(depths), max(depths)
            
            ### get rank of move
            rank = moves.index(move) if move in moves else 'X'
            
            ### get engine eval and depth on chosen move
            try:
                pvDict = infoDicts[rank]
            except:
                engine_eval = '_'
                depth_move = '_'
            else:
                engine_eval = pvDict['score']
                depth_move = pvDict['depth']
            
            ### get primary variation eval
            primary_engine_eval = infoDicts[0]['score']
            depth_best = infoDicts[0]['depth']
            
            ### get best move
            best_san = board.san(moves[0])
            
        moveDict = {'ply': ply, 'color': color, 
                    'player': player, 'elo': elo,
                    'move': san, 'rank': rank, 'eval': engine_eval, 
                    'best_move': best_san, 'best_eval': primary_engine_eval,
                    'depth_min': depth_min, 'depth_max': depth_max,
                    'depth_move': depth_move, 'depth_best': depth_best,
                    'date': date, 'event': event, 'round': game_round, 
                    'result': game_result} 
       
        moveSer = pd.Series(moveDict)
        outSeries.append(moveSer)
        
        board.push(move)
    
    engine.close()
    
    outFrame = pd.concat(outSeries, axis = 1).T
    
    elapsed = time.time() - start
    elapsed_mins = f'{round(elapsed / 60, 2)} Minutes'
    print(f'... Done with {white_player} against {black_player} at {event}! Time elapsed: {elapsed_mins}')
    
    return outFrame


def get_accuracy_frame(outFrame, player_of_interest, 
                       leniency = 0, verbose = False):
    """
    Using the outFrame supplied by get_move_evals, this function calculates
    the accuracy in each game in each round for a player_of_interest

    Parameters
    ----------
    outFrame : pd.DataFrame
        The collection of raw data supplied by get_move_evals, 
        or many concatenated instances of it.
    player_of_interest : str
        The player who we are interested in the aggregated scores for.
    leniency : float, optional
        The leniency in terms of expected game outcome, where W = 1, D = 0.5, L = 0. 
        If a move is less than {leniency} worse than the best move, it is accepted.
    verbose : Bool, optional
        Whether to print or not. The default is False.

    Returns
    -------
    pd.DataFrame
        A table with tournaments in the rows and game rounds in the columns,
        and the engine correlation percentage in the cells.
    """
    
    
    def get_pov_exp(povScore):
        if povScore == '_':
            return '_'
        else:
            w, d, l = povScore.wdl()
            return (w + d * 0.5) / (w + d + l)
        
    def get_accuracy(player_game_frame, leniency = 0):
        non_book_frame = player_game_frame[~(player_game_frame['rank'] == 'B')]
        
        non_book_frame['exp'] = non_book_frame['exp'].replace('_', 0).astype(float)
        non_book_frame['exp_loss'] = non_book_frame['exp'] - non_book_frame['best_exp']
    
        good_move = non_book_frame['exp_loss'] >= - leniency
    
        return good_move.mean()
    
    # book_move = outFrame['rank'] == 'B'
    # bad_move = outFrame['rank'] == 'X'
    # eval_move = ~ (book_move | bad_move)
    
    ### Add the Expected Outcome Values for Chosen Move and Best Move
    outFrame['exp'] = outFrame['eval'].apply(get_pov_exp)
    outFrame['best_exp'] = outFrame['best_eval'].apply(get_pov_exp)
    
    ### Separate the weird dot notation in the round numbers.
    outFrame[['round_left', 'round_right']] = outFrame['round'].str.split('.', expand = True)
    outFrame['round_left'] = outFrame['round_left'].astype(int)
    
    ### Calculate the accuracy
    accFrame = outFrame.groupby(['event', 'round_left', 'player']).apply(get_accuracy, 
                                                                         leniency = leniency)
    accFrame = accFrame.unstack('player')[player_of_interest]
    if verbose: print('Average Performance in Tournaments of Interest:')
    if verbose: print(accFrame.groupby('event').mean())

    ### Re-Shape to Tournament x Rounds
    accFrame = accFrame.unstack('round_left')
    
    return accFrame
