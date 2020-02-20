from utils import readCSV
from methods import getTradeLogStats, getSprCrossInfo
from models import getHTMLIDStrFromModelStr

import numpy as np
import pandas as pd
import keras.backend as K
import os
from bs4 import BeautifulSoup as Soup
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def createPnlVisPlot(config_dir, desired_vis_plot_fn, ts_window_size=5000):
    """
    Creates a pnl visualization plot and saves it in the Plots/ directory.

    ts_window_size determines how many timesteps to consider to determine the <worst> and <best> time windows.
    """
    # [x] find and load the trade_log
    #     filenames are pnl_vis_<model_name>_<sess_type>_<episode>_<total, best, worst>.png
    print('creating pnl vis plot:', desired_vis_plot_fn)

    #we need to first remove the model_str, then we can split by '_'
    before_model_str = desired_vis_plot_fn.split('[')[0]
    after_model_str = desired_vis_plot_fn.split(']')[-1]
    model_str = desired_vis_plot_fn.split(before_model_str)[1].split(after_model_str)[0]

    sess_type = after_model_str.split('_')[1]
    episode = after_model_str.split('_')[2]
    plot_type = after_model_str.split('_')[3].split('.')[0]
    sess_type = '' if sess_type == 'train' else sess_type #trade_log files are saved with the sesstype only if it != 'train'

    trade_log_fn = config_dir + 'Deep_Models/'+ model_str +'/Trade_Logs/'+sess_type+episode+'.csv'
    trade_logDF = readCSV(trade_log_fn)

    # [x] create plot
    full_logDF = trade_logDF.copy()
    full_logDF['Midpt'] = (full_logDF.C_B + full_logDF.C_A)/2
    full_logDF['DayPNL'] = (full_logDF.ActualAction!=0)*(full_logDF.C_B - full_logDF.Midpt) #it's always negative half the bid-ask spread
    full_logDF['OpenPNL'] = full_logDF.NewPosition.shift() * (full_logDF.Midpt - full_logDF.Midpt.shift())
    full_logDF.OpenPNL.loc[0] = 0
    full_logDF['TotalPNL'] = full_logDF.DayPNL + full_logDF.OpenPNL

    #subset the log if plotting best or worst window
    if plot_type == 'best':
        rolling_sum = full_logDF.TotalPNL.rolling(window=ts_window_size).sum()
        idx_max = rolling_sum[::-1].idxmax() # the [::-1] selects the last occurence of idxmax
        full_logDF = full_logDF[(idx_max-ts_window_size):idx_max]
    if plot_type == 'worst':
        rolling_sum = full_logDF.TotalPNL.rolling(window=ts_window_size).sum()
        idx_min = rolling_sum[::-1].idxmin() # the [::-1] selects the last occurence of idxmax
        full_logDF = full_logDF[(idx_min-ts_window_size):idx_min]

    full_logDF['CumPNL'] = full_logDF.TotalPNL.cumsum()
    # [x] graph (from http://kitchingroup.cheme.cmu.edu/blog/2013/09/13/Plotting-two-datasets-with-very-different-scales/)
    buys = full_logDF.loc[full_logDF.ActualAction == 1][['Minute', 'C_A']]
    sells = full_logDF.loc[full_logDF.ActualAction == 2][['Minute', 'C_B']]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(full_logDF.Minute, full_logDF.Midpt, 'k-', linewidth=0.5)
    ax1.plot(buys.Minute, buys.C_A, 'g+')
    ax1.plot(sells.Minute, sells.C_B, 'r+')
    ax1.set_ylabel('Price')
    # [x] add plot title
    ax1.set_title(plot_type)

    ax2 = ax1.twinx()
    ax2.plot(full_logDF.Minute, full_logDF.CumPNL, 'b-')
    ax2.set_ylabel('PNL')

    # [x] Add pnl=0 line
    # ax2.hlines(y=0, xmin=full_logDF.Minute.iloc[0] , xmax=full_logDF.Minute.iloc[-1], colors='0.75', linestyles='dashed')
    ax2.axhline(y=0, color='0.75', linestyle='dashed') # using this instead of hlines frees us from supplying xmin and xmax

    # save plot
    plt.savefig(config_dir+'Plots/'+desired_vis_plot_fn, dpi=256)

def plotProgSum(config_dir, model_str, model_progDF, x_col, y_col, scale_col,
                x_label=None, y_label=None, y_colors=None, zero_hline=True, max_point_size=60):
    """
    Plots the progress summary and saves the file.

    Inputs:
        scale_col = what to scale the scatter points by.
        x_label, y_label = If None, label by the x_col, y_col
        zero_hline = whether to place a zero line on the y axis.
        max_point_size = the maximum point size on the graph for scatter points.

    y_col can be a list of columns. If so, please supply a y_colors list of equal length and a y_label.

    Will save the file as config_dir+'Plots/'+model_str+y_label'.png'
    ... but will replace any '/' in y_label with '_over_'
    """

    if type(y_col) != str:
        assert len(y_col) == len(y_colors), 'You have entered a non-str y_col without an equal length y_colors:'+ \
            ' y_col='+str(y_col)+ ' y_colors='+str(y_colors)
        assert y_label, 'You have entered a non-str y_col without a y_label'

    if y_label is None: y_label = y_col
    if x_label is None: x_label = x_col

    sess_dict = {'train': 'green', 'eval': 'blue', 'val': 'red'}
    strat_dict = {'dqn': 'o', 't-dqn': '^', 'double-dqn': 's'}

    # [x] check that all (sess, strat) combinations cover model_progDF
    assert np.all(model_progDF.SessType.isin(sess_dict.keys())), str(model_progDF.SessType.loc[~model_progDF.SessType.isin(sess_dict.keys())].unique())+'not in sess_dict.keys()'
    assert np.all(model_progDF.Strat.isin(strat_dict.keys())), str(model_progDF.Strat.loc[~model_progDF.Strat.isin(strat_dict.keys())].unique())+'not in strat_dict.keys()'

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    if type(y_col) == str:
        ax1.plot(model_progDF[x_col], model_progDF[y_col], 'k-', linewidth=.75)
    else:
        for i, y_col_i in enumerate(y_col):
            ax1.plot(model_progDF[x_col], model_progDF[y_col_i], color=y_colors[i], linewidth=1.0)

    ax1.set_ylabel(y_label)
    ax1.set_xlabel(x_label)
    if zero_hline:
        ax1.hlines(y=0, xmin=model_progDF[x_col].iloc[0] , xmax=model_progDF[x_col].iloc[-1], colors='0.75', linestyles='dashed')

    legend_elements = []
    legend_loc = 'lower right'
    if type(y_col) == str:
        # check if the data will be at the lower right, if so, change the legend_loc to upper right
        if model_progDF[y_col].iloc[-5:].mean() < (model_progDF[y_col].max()-model_progDF[y_col].min())*.5+model_progDF[y_col].min():
            legend_loc = 'upper right'
    else:
        legend_loc = 'upper left' #may want to change this later

    for sess_type in sess_dict.keys():
        for strat in strat_dict.keys():
            prog_subDF = model_progDF.loc[(model_progDF.SessType == sess_type) & (model_progDF.Strat == strat)]
            if len(prog_subDF) > 0:
                if type(y_col) == str:
                    ax1.scatter(prog_subDF[x_col], prog_subDF[y_col], facecolors='none',
                                edgecolors=sess_dict[sess_type], marker=strat_dict[strat],
                                s=prog_subDF[scale_col]*max_point_size/model_progDF[scale_col].max(), label=sess_type+' '+strat)
                else:
                    for i, y_col_i in enumerate(y_col):
                        ax1.scatter(prog_subDF[x_col], prog_subDF[y_col_i], facecolors='none',
                                edgecolors=sess_dict[sess_type], marker=strat_dict[strat],
                                s=prog_subDF[scale_col]*max_point_size/model_progDF[scale_col].max(), label=sess_type+'_'+strat)
                legend_elements.append(Line2D([0], [0], marker=strat_dict[strat], color='none', markeredgecolor=sess_dict[sess_type], label=sess_type+' '+strat,
                          markerfacecolor='none', markersize=15))

    # [x] add legend
    ax1.legend(handles=legend_elements, loc=legend_loc)
    if type(y_col) == str:
        ax1.legend(handles=legend_elements, loc=legend_loc)
    else:
        #add additional line legend
        # Create a legend for the sess_type + strat
        first_legend = ax1.legend(handles=legend_elements, loc=legend_loc)

        # Add this legend manually to the current Axes.
        plt.gca().add_artist(first_legend)

        # Create another legend for the line colors
        line_legend_elements = [Line2D([0], [0], color=y_colors[i], label=y_col_i) for i,y_col_i in enumerate(y_col)]
        second_legend_loc = 'lower left' if legend_loc=='upper left' else 'upper left'

        plt.legend(handles=line_legend_elements, loc=second_legend_loc)

    # save plot
    plt.tight_layout()
    plt.savefig(config_dir+'Plots/'+model_str+'_'+y_label.replace('/', '_over_')+'.png', dpi=256)

def createModelProgressSummary(config_dir, model_str):
    """
    Creates plots for the model, showing how different metrics have progressed over time.
    Delineate between SessTypes, Strategies
    Change size of point based on how many TsSeen
    Returns Soup object.
    """
    all_prog_fn = config_dir + 'Deep_Models/all_progress_summary.csv'
    all_progDF = pd.read_csv(all_prog_fn)

    # create necessary columns
    model_progDF = all_progDF.loc[all_progDF.Model == model_str].reset_index(drop=True)
    model_progDF['CumTsSeen'] = model_progDF.TsSeen.cumsum()
    model_progDF['CumLossUpd'] = model_progDF.LossUpd.cumsum()
    model_progDF['CumTargetUpd'] = model_progDF.TargetUpd.cumsum()
    model_progDF['AvgTsSeen'] =  model_progDF.CumTsSeen - (model_progDF.TsSeen/2)
    model_progDF['AvgLossUpd'] =  model_progDF.CumLossUpd - (model_progDF.LossUpd/2)
    model_progDF['AvgTargetUpd'] =  model_progDF.CumTargetUpd - (model_progDF.TargetUpd/2)
    model_progDF['PNLpTS'] =  model_progDF.PNL/model_progDF.TsSeen
    model_progDF['NumTradespTS'] =  model_progDF.NumTrades/model_progDF.TsSeen
    epsilon = 1e-8 #to avoid 0 --> -Inf
    model_progDF['AvgLogLoss'] =  np.log10(model_progDF.AvgLoss + epsilon)

    # [x] create and save the relevant plots
    # PNLpDay / TsSeen
    plotProgSum(config_dir, model_str, model_progDF=model_progDF,
                x_col='AvgTsSeen', y_col='PNLpTS', scale_col='TsSeen',
                y_label='PNL/TsSeen')

    # AvgLogLoss / LossUpdate
    plotProgSum(config_dir, model_str, model_progDF=model_progDF,
                x_col='AvgLossUpd', y_col='AvgLogLoss', scale_col='LossUpd', zero_hline=False)

    # NumTradespTS / TsSeen
    plotProgSum(config_dir, model_str, model_progDF=model_progDF,
                x_col='AvgTsSeen', y_col='NumTradespTS', scale_col='TsSeen',
                y_label='NumTrades/TsSeen')

    # LrgSprPct / LossUpdate
    plotProgSum(config_dir, model_str, model_progDF=model_progDF,
                x_col='AvgLossUpd', y_col='LrgSprPct', scale_col='LossUpd')

    # LrgSprLikely / LossUpdate
    plotProgSum(config_dir, model_str, model_progDF=model_progDF,
                x_col='AvgLossUpd', y_col='LrgSprLikely', scale_col='LossUpd')

    # LrgSprPref / LossUpdate
    plotProgSum(config_dir, model_str, model_progDF=model_progDF,
                x_col='AvgLossUpd', y_col='LrgSprPref', scale_col='LossUpd')

    # Pct <Flat, Long, Short> / LossUpdate
    plotProgSum(config_dir, model_str, model_progDF=model_progDF,
                x_col='AvgLossUpd',
                y_col=['PctFlat', 'PctLong', 'PctShort'],
                y_colors=['blue', 'green', 'red'],
                y_label='PositionPct',
                scale_col='LossUpd', zero_hline=True)

    # Sharpe / LossUpdate
    plotProgSum(config_dir, model_str, model_progDF=model_progDF,
                x_col='AvgLossUpd', y_col='Sharpe', scale_col='LossUpd')

    # [x] create soup object for the html and return

    # <div id=model_str_html>
    #   <h3> model_str </h3>
    #   <table> first_3, mid_3, last_3 </table>
    #   <img src= > ...etc

    model_prog_soup = Soup()
    model_prog_div = model_prog_soup.new_tag('div', id=getHTMLIDStrFromModelStr(model_str))
    model_h3 = model_prog_soup.new_tag('h3')
    model_h3.string = model_str
    model_prog_div.append(model_h3)

    # create first, mid, last table
    n_epi_samples = 2 # will take the first, mid, and last n_epi_samples episodes
    fmlDF = model_progDF
    n_epis = len(model_progDF)
    if n_epis > n_epi_samples*3:
        fmlDF = fmlDF[0:0]
        fmlDF = fmlDF.append(model_progDF.iloc[:n_epi_samples])
        fmlDF = fmlDF.append(model_progDF.iloc[int(n_epis/2-n_epi_samples/2):int(n_epis/2-n_epi_samples/2)+n_epi_samples])
        fmlDF = fmlDF.append(model_progDF.iloc[-n_epi_samples:])
    fml_table_soup = Soup(fmlDF.to_html(col_space=50, justify='center'))
    fml_table_soup.table['style'] = 'white-space: nowrap'
    model_prog_div.append(fml_table_soup)

    # [x] handle images
    # PNL/TsSeen
    model_prog_div.append(model_prog_soup.new_tag('img', height='400',
                            src='Plots/'+model_str+'_'+'PNL_over_TsSeen'+'.png'))

    # AvgLogLoss
    model_prog_div.append(model_prog_soup.new_tag('img', height='400',
                            src='Plots/'+model_str+'_'+'AvgLogLoss'+'.png'))

    # NumTrades/TsSeen
    model_prog_div.append(model_prog_soup.new_tag('img', height='400',
                            src='Plots/'+model_str+'_'+'NumTrades_over_TsSeen'+'.png'))

    model_prog_div.append(model_prog_soup.new_tag('br'))
    # LrgSprPct
    model_prog_div.append(model_prog_soup.new_tag('img', height='400',
                            src='Plots/'+model_str+'_'+'LrgSprPct'+'.png'))

    # LrgSprLikely
    model_prog_div.append(model_prog_soup.new_tag('img', height='400',
                            src='Plots/'+model_str+'_'+'LrgSprLikely'+'.png'))

    # LrgSprPref
    model_prog_div.append(model_prog_soup.new_tag('img', height='400',
                            src='Plots/'+model_str+'_'+'LrgSprPref'+'.png'))

    model_prog_div.append(model_prog_soup.new_tag('br'))
    # PositionPct
    model_prog_div.append(model_prog_soup.new_tag('img', height='400',
                            src='Plots/'+model_str+'_'+'PositionPct'+'.png'))

    # Sharpe
    model_prog_div.append(model_prog_soup.new_tag('img', height='400',
                            src='Plots/'+model_str+'_'+'Sharpe'+'.png'))

    return model_prog_div

def findMod(bigger_soup, elem_type, search_string):
    """
    Guards against not finding the element in find() because of newlines
    """
    elems = bigger_soup.find_all(elem_type)
    for elem in elems:
        if search_string in elem.string:
            return elem
    return None

def updateSummarySheet(config_dir):
    print('in updateSummarySheet')
    summary_sheet_fn = config_dir+'summary_sheet.html'
    with open(summary_sheet_fn,'r') as fh:
        summary_sheet_html = fh.read()
    summary_soup = Soup(summary_sheet_html)
    all_progress_summary_fn = config_dir + 'Deep_Models/all_progress_summary.csv'
    if not os.path.exists(all_progress_summary_fn):
        print('all_progress_summary_fn does not exist in path:', all_progress_summary_fn, '...returning')
        return
    progressDF = pd.read_csv(all_progress_summary_fn)
    # [x] check that there is sufficient data to publish best_models_table
    bestDF = progressDF.copy()
    bestDF = bestDF.loc[bestDF.TsSeen >= 14999]

    if len(bestDF) == 0:
        print('not enough data for bestDF')
    else:
        model_strs = bestDF.Model.unique()
        # [x] get the top 3 models for each model_str
        # I'm sure there is a more elegant pandas way to do this
        best3DF = bestDF[-1:0]
        for model_str in model_strs:
            model_subDF = bestDF.loc[bestDF.Model == model_str]
            model_pnl_3rd_best = model_subDF.PNLpDay.nlargest(n=3).iloc[-1]
            best3DF = best3DF.append(model_subDF.loc[model_subDF.PNLpDay >= model_pnl_3rd_best]) #append the rows which have at least as good pnl as the 3rd best for that model_str
        best3DF.sort_values('PNLpDay', ascending=False, inplace=True)
        # update best_models_summary
        best_models_table = summary_soup.select('#best_models_table')
        best_models_html = best3DF.to_html(table_id = 'best_models_table', index=False, col_space=50, justify='center')
        best_models_soup = Soup(best_models_html)
        # [x] change the model names to link to the model sections
        tbody_soup = best_models_soup.tbody
        trs = tbody_soup.find_all('tr')
        for tr in trs:
            first_td = tr.find_all('td')[0] #this is the td with the model name
            td_model_str = '['+']'.join('['.join(first_td.text.split('[')[1:]).split(']')[:-1])+']'
            a_tag = summary_soup.new_tag('a', href='#'+getHTMLIDStrFromModelStr(td_model_str))
            a_tag.string = first_td.text
            first_td.string = ''
            first_td.append(a_tag)
        best_models_soup.table['style'] = 'white-space: nowrap'
        if len(best_models_table) == 0:
            print('No best_models_table found. Creating one.')
            best_models_h1 = findMod(summary_soup, 'h1', 'Best Models Summary')
            best_models_h1.insert_after(best_models_soup)
        else:
            best_models_table = best_models_table[0]
            best_models_table.replace_with(best_models_soup)

        # Get the best 1 model for each model_str
        # I'm sure there is a more elegant pandas way to do this
        best1DF = bestDF[-1:0]
        for model_str in model_strs:
            model_subDF = bestDF.loc[bestDF.Model == model_str]
            model_pnl_max = model_subDF.PNLpDay.max()
            best1DF = best1DF.append(model_subDF.loc[model_subDF.PNLpDay == model_pnl_max].iloc[0]) #append the first row which has the max pnl for that model_str
        best1DF.sort_values('PNLpDay', ascending=False, inplace=True)
        # make sure that the model/episode/sesstype combo is the most recent in progressDF
        for i in best1DF.index:
            progressDFsub = progressDF.loc[(progressDF.Model == best1DF.loc[i].Model) & (progressDF.Epi == best1DF.loc[i].Epi) & (progressDF.SessType == best1DF.loc[i].SessType)]
            last_i = progressDFsub.index[-1]
#             print(i)
#             print(best1DF.loc[i].Model)
#             print(best1DF.loc[i].Epi)
#             print(best1DF.loc[i].SessType)
#             print(progressDFsub)
#             print(best1DF)

            assert i == last_i, 'Problem with '+best1DF.loc[i].Model+': '+str(last_i)+' is more recent than '+str(i)

        # [x] update best pnl visualizations
        # you can identify the trade_logs by [Epi, Model, SessType]
        # [x] check what plots we have in the plots folder.
        #     filenames are pnl_vis_<model_name>_<sess_type>_<episode>_<total, best, worst>.png
        plots_dir = config_dir + 'Plots/'
        current_plot_filenames = [f for f in os.listdir(plots_dir) if f[:7]=='pnl_vis']
        needed_plot_filenames = []
        for i in best1DF.index:
            for plot_type in ['total', 'best', 'worst']:
                needed_plot_filenames.append('pnl_vis_'+best1DF.loc[i].Model+'_'+best1DF.loc[i].SessType+'_'+
                                             str(best1DF.loc[i].Epi)+'_'+plot_type+'.png')
        extraneous_current_plot_filenames = set(current_plot_filenames).difference(set(needed_plot_filenames))
        needed_plot_filenames = set(needed_plot_filenames).difference(set(current_plot_filenames))
        if len(needed_plot_filenames) != 0:
            print('needed_plot_filenames:', needed_plot_filenames, 'extraneous_current_plot_filenames:', extraneous_current_plot_filenames)
        # create new necessary pnl vis plots
        for fn in needed_plot_filenames:
            try:
                createPnlVisPlot(config_dir, fn)
            except Exception as e:
                print('Error! :')
                print(e)
                pass

        # delete unneeded pnl vis files
        for fn in extraneous_current_plot_filenames:
            os.remove(plots_dir+fn)
        # [x] change the html to reflect the new plots
        new_best_models_div = summary_soup.new_tag('div', id='best_models_div')
        for i in best1DF.index:
            # wrap a link in h3 so it links to the Model Progress Summary
            model_h3 = summary_soup.new_tag('h3')
            model_a = summary_soup.new_tag('a', href='#'+getHTMLIDStrFromModelStr(best1DF.loc[i].Model))
            model_a.string = best1DF.loc[i].Model
            model_h3.append(model_a)
            model_table = Soup(best1DF.loc[i:i].to_html(col_space=50, justify='center'))
            model_table.table['style'] = 'white-space: nowrap'
            model_total_pnl_vis_plot = summary_soup.new_tag('img', height='400',
                    src='Plots/'+'pnl_vis_'+best1DF.loc[i].Model+'_'+best1DF.loc[i].SessType+'_'+
                        str(best1DF.loc[i].Epi)+'_total.png')
            model_worst_pnl_vis_plot = summary_soup.new_tag('img', height='400',
                    src='Plots/'+'pnl_vis_'+best1DF.loc[i].Model+'_'+best1DF.loc[i].SessType+'_'+
                        str(best1DF.loc[i].Epi)+'_worst.png')
            model_best_pnl_vis_plot = summary_soup.new_tag('img', height='400',
                    src='Plots/'+'pnl_vis_'+best1DF.loc[i].Model+'_'+best1DF.loc[i].SessType+'_'+
                        str(best1DF.loc[i].Epi)+'_best.png')

            new_best_models_div.append(model_h3)
            new_best_models_div.append(model_table)
            new_best_models_div.append(model_total_pnl_vis_plot)
            new_best_models_div.append(model_worst_pnl_vis_plot)
            new_best_models_div.append(model_best_pnl_vis_plot)

        best_models_div = summary_soup.select('#best_models_div')
        if len(best_models_div) == 0:
            print('No best_models_div currently exists. Inserting our newly created one.')
            model_progress_summary_h1 = findMod(summary_soup, 'h1', 'Model Progress Summary')
            model_progress_summary_h1.insert_before(new_best_models_div)
        else:
            best_models_div = best_models_div[0]
            best_models_div.replace_with(new_best_models_div)

        # [x] update model_progress_summary
        for model_str in model_strs:
            # create model progress graphs
            new_model_prog_div = createModelProgressSummary(config_dir, model_str)
            old_model_prog_div = summary_soup.select('#'+getHTMLIDStrFromModelStr(model_str))
            if len(old_model_prog_div) == 0:
                print('\nNo old_model_prog_div found for ', model_str, 'Creating now')
                progress_details_h1 = findMod(summary_soup, 'h1', 'Progress Details')
                progress_details_h1.insert_before(new_model_prog_div)
            else:
                old_model_prog_div = old_model_prog_div[0]
                old_model_prog_div.replace_with(new_model_prog_div)
    # [NA] update progress_details
    # NA for now
    print('Rewriting html to '+summary_sheet_fn)
    with open(summary_sheet_fn,'w') as fh:
        fh.write(summary_soup.prettify())

def saveProgressSummary(agent, dataObj, batch_size, epsilon_start, epsilon_end, avg_loss_array, session_type, start_time, end_time):
    pnl, pnl_per_day, sharpe, num_trades, drawdown, drawup, \
        ts_seen, pct_flat, pct_long, pct_short, data_start, data_end = getTradeLogStats(agent)

    lrg_spr_cross_pct, lrg_spr_cross_likelihood, lrg_spr_hold_preference = getSprCrossInfo(agent, dataObj)

    new_row_dict = {'Epi': agent.episode,
                    'SessType': session_type,
                    'Strat': agent.strategy,
                    'PNL': pnl,
                    'PNLpDay': pnl_per_day,
                    'Sharpe': sharpe,
                    'AvgLoss': np.mean(avg_loss_array),
                    'NumTrades': num_trades,
                    'DrawDown': drawdown,
                    'DrawUp': drawup,
                    'LrgSprPct': lrg_spr_cross_pct,
                    'LrgSprLikely': lrg_spr_cross_likelihood,
                    'LrgSprPref': lrg_spr_hold_preference,
                    'TimeTakenMin': (end_time-start_time).seconds/60,
                    'TsSeen': ts_seen,
                    'PctFlat': pct_flat,
                    'PctLong': pct_long,
                    'PctShort': pct_short,
                    'DataStart': data_start,
                    'DataEnd': data_end,
                    'LossUpd': agent.num_loss_updates,
                    'TargetUpd': agent.num_target_updates,
                    'Gamma': agent.gamma,
                    'EpsilonStart': epsilon_start,
                    'EpsilonEnd': epsilon_end,
                    'EpsilonDecay': agent.epsilon_decay,
                    'ResetEvery': agent.reset_every,
                    'LossType': str(agent.loss).split('function ')[1].split(' ')[0].split('_loss')[0],
                    'LossParam': np.nan, #maybe should actually do this later
                    'OptType': str(agent.model.optimizer.__class__).split('.')[-1].replace('\'', '').replace('>', ''), #https://stackoverflow.com/questions/49785536/get-learning-rate-of-keras-model
                    'OptLR': K.eval(agent.model.optimizer.lr),
                    'StartTime': start_time,
                    'EndTime': end_time,
                    'CrossPrem': agent.cross_q_prem,
                    'FlatPrem': agent.flat_q_prem
                    }

    round_info = {'PNL': 2, 'PNLpDay': 2, 'Sharpe': 3, 'AvgLoss': 5, 'DrawDown': 2, 'DrawUp': 2,
                  'LrgSprPct': 3, 'LrgSprLikely': 4, 'LrgSprPref': 5, 'TimeTakenMin': 0,
                  'PctFlat': 2, 'PctLong': 2, 'PctShort': 2,
                  'EpsilonStart': 2, 'EpsilonEnd': 2, 'EpsilonDecay': 3,
                  'CrossPrem': 4, 'FlatPrem': 4}

    col_order = ['Epi', 'SessType', 'Strat',
                'PNL', 'PNLpDay', 'Sharpe', 'AvgLoss', 'NumTrades',
                'DrawDown', 'DrawUp', 'LrgSprPct', 'LrgSprLikely', 'LrgSprPref',
                'TimeTakenMin', 'TsSeen', 'PctFlat', 'PctLong', 'PctShort',
                'DataStart', 'DataEnd', 'LossUpd', 'TargetUpd',
                'Gamma', 'EpsilonStart', 'EpsilonEnd', 'EpsilonDecay', 'ResetEvery',
                'LossType', 'LossParam', 'OptType', 'OptLR', 'StartTime', 'EndTime',
                'CrossPrem', 'FlatPrem']

    assert set(col_order) == set(new_row_dict.keys())
    assert len(set(round_info.keys()).difference(set(new_row_dict.keys()))) == 0, set(round_info.keys()).difference(set(new_row_dict.keys()))

    progress_summary_fn = agent.model_dir+'progress_summary.csv'
    progress_summaryDF = pd.DataFrame(new_row_dict, index=[0])
    if os.path.exists(progress_summary_fn):
        progress_summaryDF = pd.read_csv(progress_summary_fn)
        progress_summaryDF = progress_summaryDF.append(new_row_dict, ignore_index=True)

    progress_summaryDF = progress_summaryDF.round(round_info)
    progress_summaryDF[col_order].to_csv(progress_summary_fn, index=False)

    all_progress_summary_fn = agent.config_dir+'Deep_Models/all_progress_summary.csv'
    col_order = ['Model'] + col_order
    new_row_dict['Model'] = agent.model_cfg.model_str

    all_progress_summaryDF = pd.DataFrame(new_row_dict, index=[0])

    if os.path.exists(all_progress_summary_fn):
        all_progress_summaryDF = pd.read_csv(all_progress_summary_fn)
        all_progress_summaryDF = all_progress_summaryDF.append(new_row_dict, ignore_index=True)

    all_progress_summaryDF = all_progress_summaryDF.round(round_info)
    all_progress_summaryDF[col_order].to_csv(all_progress_summary_fn, index=False)
    print_keys = ['PNL', 'PNLpDay', 'Sharpe', 'AvgLoss', 'NumTrades', 'DrawDown', 'DrawUp', 'LrgSprPct', 'LrgSprLikely', 'LrgSprPref', 'TimeTakenMin', 'TsSeen', 'PctFlat', 'PctLong', 'PctShort', 'LossUpd', 'TargetUpd', 'EpsilonStart', 'EpsilonEnd']
    print('\t\t'.join([key+': '+str(new_row_dict[key]) for key in print_keys]))
