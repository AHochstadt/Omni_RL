omni_dir = '/home/andrew/All_Trading/Studies/Omni_Project/'
tdm_dir = '/media/andrew/FreeAgent Drive/Market_Data/Tick_Data_Manager/'

day_chg_incs = [1, 3, 5] # number of days to calculate for day_cols
minute_incs = [1, 5, 15, 30, 60] # number of minutes in the past to calculate for minute cols

action_size = 3 # HOLD, BUY, SELL

keras_models = [

    [{'layer': 'Dense', 'activation': 'relu', 'units': 256},
     {'layer': 'Dense', 'activation': 'relu', 'units': 128},
     {'layer': 'Dense', 'activation': 'relu', 'units': 64},
     {'layer': 'Dense', 'activation': 'relu', 'units': 64},
     {'layer': 'Dense', 'activation': 'relu', 'units': 64},
     {'layer': 'Dense', 'activation': 'linear', 'units': action_size}],

    [{'layer': 'Dense', 'activation': 'relu', 'units': 256},
     {'layer': 'Dense', 'activation': 'relu', 'units': 128},
     {'layer': 'Dropout', 'rate': 0.2},
     {'layer': 'Dense', 'activation': 'relu', 'units': 64},
     {'layer': 'Dense', 'activation': 'relu', 'units': 64},
     {'layer': 'Dense', 'activation': 'relu', 'units': 64},
     {'layer': 'Dense', 'activation': 'linear', 'units': action_size}],

    [{'layer': 'Dense', 'activation': 'relu', 'units': 256},
     {'layer': 'Dense', 'activation': 'relu', 'units': 128},
     {'layer': 'Dropout', 'rate': 0.1},
     {'layer': 'Dense', 'activation': 'relu', 'units': 64},
     {'layer': 'Dense', 'activation': 'relu', 'units': 64},
     {'layer': 'Dense', 'activation': 'relu', 'units': 64},
     {'layer': 'Dense', 'activation': 'linear', 'units': action_size}],

    [{'layer': 'Dense', 'activation': 'relu', 'units': 256},
     {'layer': 'Dense', 'activation': 'relu', 'units': 128},
     {'layer': 'Dropout', 'rate': 0.2},
     {'layer': 'Dense', 'activation': 'relu', 'units': 64},
     {'layer': 'Dense', 'activation': 'relu', 'units': 64},
     {'layer': 'Dropout', 'rate': 0.2},
     {'layer': 'Dense', 'activation': 'relu', 'units': 64},
     {'layer': 'Dense', 'activation': 'linear', 'units': action_size}],

    [{'layer': 'Dense', 'activation': 'relu', 'units': 256},
     {'layer': 'Dense', 'activation': 'relu', 'units': 128},
     {'layer': 'Dropout', 'rate': 0.2},
     {'layer': 'Dense', 'activation': 'relu', 'units': 64},
     {'layer': 'Dense', 'activation': 'relu', 'units': 64},
     {'layer': 'Dropout', 'rate': 0.2},
     {'layer': 'Dense', 'activation': 'relu', 'units': 64},
     {'layer': 'Dense', 'activation': 'relu', 'units': 64},
     {'layer': 'Dropout', 'rate': 0.2},
     {'layer': 'Dense', 'activation': 'relu', 'units': 64},
     {'layer': 'Dense', 'activation': 'relu', 'units': 64},
     {'layer': 'Dropout', 'rate': 0.2},
     {'layer': 'Dense', 'activation': 'relu', 'units': 64},
     {'layer': 'Dense', 'activation': 'linear', 'units': action_size}],


    [{'layer': 'Dense', 'activation': 'relu', 'units': 64},
     {'layer': 'Dense', 'activation': 'linear', 'units': action_size}],

    [{'layer': 'Dense', 'activation': 'relu', 'units': 256},
     {'layer': 'Dense', 'activation': 'linear', 'units': action_size}],

    [{'layer': 'Dense', 'activation': 'relu', 'units': 128},
     {'layer': 'Dense', 'activation': 'relu', 'units': 64},
     {'layer': 'Dense', 'activation': 'linear', 'units': action_size}],

    [{'layer': 'Dense', 'activation': 'relu', 'units': 256},
     {'layer': 'Dense', 'activation': 'relu', 'units': 128},
     {'layer': 'Dense', 'activation': 'linear', 'units': action_size}],

    [{'layer': 'Dense', 'activation': 'relu', 'units': 256},
     {'layer': 'Dense', 'activation': 'relu', 'units': 128},
     {'layer': 'Dense', 'activation': 'relu', 'units': 64},
     {'layer': 'Dense', 'activation': 'linear', 'units': action_size}]
]
