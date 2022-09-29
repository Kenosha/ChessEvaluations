# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 01:41:09 2022

@author: gianl
"""


import chess
import chess.pgn
import chess.engine
import chess.polyglot

import funcs as cf 

import os


##### DATA AND ENGINE SETTINGS #####
pgnStr = 'Niemann.pgn'
engineExe = 'stockfish 7 x64 bmi2.exe'
openingStr = 'codekiddy.bin'

######## PARAMETERS ###########
PLAYER_NAME = 'Niemann, Hans Moke'
SAVE_RESULTS_AS = 'Niemann'

ENGINE_LIMIT = 0.004 ### Time per Move
ENGINE_LINES = 10    ### Engine Lines To Calculate
WORKERS = 12         ### Number of Parallel Workers
LENIENCY = 0         ### The leniency in terms of expected game outcome, 
                       # where W = 1, D = 0.5, L = 0. 
                       # If a move is less than {leniency} worse 
                       # than the best move, it is accepted.

########## GET OPENING BOOK ########
book = chess.polyglot.open_reader(os.path.join(cf.openingPath, openingStr))

######### GET GAMES #############
eventList = ['Philadelphia', 'Annual World Open', 'Junior 2021', 
             'US Open', 'Tras-os-Montes', 'National Open']
offsetList = []
for event in eventList:
    offsets = cf.get_offsets(pgnStr, {'Event': event})
    print(f'{event}: Available Games: {len(offsets)}')
    offsetList.extend(offsets)

games = cf.read_offsets(pgnStr, offsetList)    

###### SET UP MULTIPROCESSING ######
from multiprocessing.pool import ThreadPool as Pool

from functools import partial
my_parallel_func = partial(cf.get_move_evals, book = book, engine = engineExe,
                           engine_limit = ENGINE_LIMIT, engine_lines = ENGINE_LINES)


####### ANALYSE GAMES ########
if __name__ == '__main__':
    
    ### Run Parallel Workers
    with Pool(WORKERS) as p:
        outFrames = p.map(my_parallel_func, games)

    ### Join Parallel Outputs
    import pandas as pd         
    outFrame = pd.concat(outFrames, ignore_index = True)
    
    ### Make Timestamp
    import datetime as dt
    t = dt.datetime.now()
    timestamp = f'{t.month}_{t.day}_{t.hour}_{t.minute}_{t.second}'
    saveName = f'{SAVE_RESULTS_AS}_{timestamp}.csv'
    
    ### Save Raw
    outFrame.to_csv(os.path.join(cf.savePath_raw, saveName))
    
    ### Aggregate to make the Engine Correlation Table
    accFrame = cf.get_accuracy_frame(outFrame, PLAYER_NAME, 
                                     leniency = LENIENCY)

    ### Save Aggregate
    accFrame.to_csv(os.path.join(cf.savePath_agg, saveName))
    
    ### Other Interesting Things to Output for Analytics
    # a) check depth
    avg_depth = outFrame.loc[outFrame.depth_move != '_', 'depth_move'].mean()

    # b) check average number of book moves
    avg_book = outFrame[outFrame['rank'] == 'B'].groupby(['event', 'round_left']).ply.max().mean()
    
    print('__________')
    print(f'Average Depth of Engine Calculation: {avg_depth} half moves.')
    print(f'Average Depth of Opening Book: {avg_book} half moves.')  
    

