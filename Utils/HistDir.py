# -*- coding: utf-8 -*-

##################################################################################
## Returns Directional Histogram / Wind Rose and Joint Ocurrence Table with Values
##################################################################################
## Version: 3.0.1
##################################################################################

import glob
import os
import sys
import xlsxwriter
from numpy import asarray, max, arange, round, insert, radians
from numpy import ceil, ma, cumsum, array, argmin
from numpy import linspace, meshgrid, histogram2d, flipud, size, sum
from numpy import nanmax, nanmean, nansum
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from scipy.interpolate import griddata
from scipy import interpolate, ndimage
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns

sns.set_style("whitegrid", {'grid.color': "black",
                            'axes.edgecolor': "black",
                            }
              )
sns.set_context('notebook', font_scale=1.7)


def HistDir(P, D, arqname='', Pmax=None, MaxProb=None, par='hs', MyRange=[],
            dir16=True, interpolado=True, r_angle=None, porcentagem=False, conv_oc=False):
    '''
    Makes a directional histogram for a single point

    P = array/DataFrame
        Intensity
    D = array/DataFrame
        Direction in degrees
    arqname =  string (optional, default '')
        Output file name
    Pmax =  None or float (optional, default: None)
        Fix the intensity axis ONLY for the histogram. 
        Otherwise, _None_ calculates automatically
    MaxProb = None or int (optional, default: None)
        Fix the probability axis. 
        Otherwise, _None_ calculates automatically
    par = string (optional, default 'hs')
        Set the parameter being analyzed
    MyRange = list of float (optional)
        List of parameter intensity ranges for the plot and table. 
        If left empty it will be calculated automatically
    dir16 = bool, default: True (optional, default: 16 bins)
        Whether the directions will be divided into 16 or 8 bins
    interpolado = bool, default: True (optional, default: histogram (True))
        Defines if you want to generate an interpolated histogram or wind rose
    r_angle = None or float (optional, default: None)
        If you enter a float it changes the position of the angle labels on the plot
    porcentagem = bool, default: False (optional)
        Defines if the table will be presented as a percentage or as occurrences
    conv_oc = bool, default: False (optional)
       True if you want to turn the parameters that are usually presented in meteorological convention to oceanographic convention.
       USED on [hs, tp, wind and wave energy]

    Details:
        parameters - choose depending on the analysis
                hs = Significant height
                tp = Wave period
                current = current
                wind = wind
                energy = wave energy

    output:
        A png figure and excel table with directional histogram analysis
    '''

    # transforms inputs to numpy arrays
    P = asarray(P)
    D = asarray(D)
    # check for possible user error
    if len(D.shape) > 1:
        if D.shape[1] == 1:
            D = D[:, 0]
        elif D.shape[0] == 1:
            D = D[0, :]
        else:
            print(u'Check the Direction Dimensions')
            sys.pause('')
    if len(P.shape) > 1:
        if P.shape[1] == 1:
            P = P[:, 0]
        elif P.shape[0] == 1:
            P = P[0, :]
        else:
            print(u'Check the dimensions of the input parameter')
    # checks if the maximum value is set
    if Pmax is None:
        Pmax = np.nanmax(P)
    # checks the parameter for establishing the input data
    if par == 'hs':
        # plot title
        if conv_oc:
            titulo2 = u'Wave height (m) - oceanographic convention'
        else:
            titulo2 = u'Wave height (m) - meteorological convention'
        # part of the joint occurrence table header
        cabecalho = u'Height (m)'
        # part of the output filename
        fname = arqname + '_Height'
        myfmt = '0.0'
        # sets the number of divisions of the parameter in the plot (P_bins) and occurrence table (T_bins)
        # checks if there are 5 to 10 classes
        P_bins = arange(0, round(Pmax) + 0.5, 0.5)
        if len(P_bins) > 11:
            P_bins = arange(0, round(Pmax) + 1, 1)
        T_bins = arange(0, round(np.nanmax(P)) + 0.5, 0.5)
        if len(T_bins) > 11:
            T_bins = arange(0, round(np.nanmax(P)) + 1, 1)
    elif par == 'tp':
        if conv_oc:
            titulo2 = u'Wave period (s' + u') - oceanographic convention'
        else:
            titulo2 = u'Wave period (s' + u') - meteorological convention'
        cabecalho = u'Period (s)'
        fname = arqname + '_Period'
        myfmt = '0.0'
        P_bins = arange(0, round(Pmax) + 2, 2)
        if len(P_bins) > 11:
            P_bins = arange(0, round(Pmax) + 3, 3)
        elif len(P_bins) < 6:
            P_bins = arange(0, round(Pmax) + 1, 1)
        T_bins = arange(0, round(np.nanmax(P)) + 2, 2)
        if len(T_bins) > 11:
            T_bins = arange(0, round(np.nanmax(P)) + 3, 3)
        elif len(T_bins) < 6:
            T_bins = arange(0, round(np.nanmax(P)) + 1, 1)
    elif par == 'wind':
        if conv_oc:
            titulo2 = u'- oceanographic convention'
        else:
            titulo2 = u'- meteorological convention'
        cabecalho = u'Vel. (m/s)'
        fname = arqname + '_Wind'
        myfmt = '0.0'
        P_bins = arange(0, Pmax + 2, 2)
        if len(P_bins) > 11:
            if len(P_bins) > 11:
                P_bins = arange(0, Pmax + 2.5, 2.5)
                if len(P_bins) > 11:
                    P_bins = arange(0, Pmax + 5, 5)
        elif len(P_bins) < 6:
            P_bins = arange(0, 10. + 1, 1)
            if len(P_bins) < 6:
                P_bins = arange(0, Pmax + 0.5, 0.5)
        T_bins = arange(0, np.nanmax(P) + 2, 2)
        if len(T_bins) > 11:
            if len(T_bins) > 11:
                T_bins = arange(0,  np.nanmax(P) + 2.5, 2.5)
                if len(T_bins) > 11:
                    T_bins = arange(0,  np.nanmax(P) + 5, 5)
        elif len(T_bins) < 6:
            T_bins = arange(0, 10. + 1, 1)
            if len(T_bins) < 6:
                T_bins = arange(0,  np.nanmax(P) + 0.5, 0.5)
    elif par == 'energy':
        if conv_oc:
            titulo2 = u'Wave energy (kJ/m²) - oceanographic convention'
        else:
            titulo2 = u'Wave energy (kJ/m²) - meteorological convention'
        cabecalho = u'Energy (kJ/m²)'
        fname = arqname + '_Energy'
        myfmt = '0.0'
        P_bins = arange(0, round(Pmax / 10.) * 10. + 2, 2)
        if len(P_bins) > 11:
            P_bins = arange(0, round(Pmax / 10.) * 10. + 3, 3)
        if len(P_bins) > 11:
            P_bins = arange(0, round(Pmax / 10.) * 10. + 5, 5)
        elif len(P_bins) < 6:
            P_bins = arange(0, round(Pmax / 10.) * 10. + 1, 1)
        T_bins = arange(0, round(np.nanmax(P) / 10.) * 10. + 2, 2)
        if len(T_bins) > 11:
            T_bins = arange(0, round(np.nanmax(P) / 10.) * 10. + 3, 3)
        if len(T_bins) > 11:
            T_bins = arange(0, round(np.nanmax(P) / 10.) * 10. + 5, 5)
        elif len(T_bins) < 6:
            T_bins = arange(0, round(np.nanmax(P) / 10.) * 10. + 1, 1)
    elif par == 'current':
        titulo2 = u'Current Strength (m/s) - oceanographic convention'
        cabecalho = u'Current(m/s)'
        fname = arqname + '_Current'
        myfmt = '0.00'
        P_bins = arange(0, round(Pmax * 10.) / 10. + 0.1, 0.1)
        if len(P_bins) > 11:
            if len(P_bins) > 11:
                P_bins = arange(0, round(Pmax * 10.) / 10. + 0.2, 0.2)
                if len(P_bins) > 11:
                    P_bins = arange(0, round(Pmax * 10.) / 10. + 0.25, 0.25)
        elif len(P_bins) < 6:
            P_bins = arange(0, round(Pmax * 10) / 10. + 0.05, 0.05)
            if len(P_bins) < 6:
                P_bins = arange(0, round(Pmax * 10) / 10. + 0.03, 0.03)
        T_bins = arange(0, round(np.nanmax(P) * 10.) / 10. + 0.1, 0.1)
        if len(T_bins) > 11:
            if len(T_bins) > 11:
                T_bins = arange(0, round(np.nanmax(P) * 10.) / 10. + 0.2, 0.2)
                if len(T_bins) > 11:
                    T_bins = arange(0, round(np.nanmax(P) * 10.) / 10. + 0.25, 0.25)
        elif len(T_bins) < 6:
            T_bins = arange(0, round(np.nanmax(P) * 10) / 10. + 0.05, 0.05)
            if len(P_bins) < 6:
                T_bins = arange(0, round(np.nanmax(P) * 10) / 10. + 0.03, 0.03)
    else:
        print(u'Input parameter not defined.\nSet one of the parameters:\n hs = Significant height\n tp = Wave period\n current = current\n wind = wind\n energy = wave energy')
    # Fix P_bins
    P_bins=np.round(P_bins,2)
    T_bins=np.round(T_bins,2)
    # if the user has defined the range
    if MyRange != []:
        del P_bins
        del T_bins
        P_bins = arange(0, Pmax + MyRange, MyRange)
        P_bins=np.round(P_bins,2)
        T_bins = arange(0, P.max() + MyRange, MyRange)
        T_bins=np.round(T_bins,2)
    # check if it will be 16 or 8 directions (applies to the table)
    if dir16 is True:
        # subtract from direction
        dirdiff = 11.25
        # number of directional bins
        dirrg = 17
        # table directions header
        head = [cabecalho, 'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', u'(%)']
    else:
        dirdiff = 22.5
        dirrg = 9
        head = [cabecalho, 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', u'(%)']
    # Polar plot
    ax = plt.subplot(1, 1, 1, polar=True)
    # defines that zero will be in the north
    ax.set_theta_zero_location("N")
    # runs the azimuital graph
    ax.set_theta_direction(-1)
    ax.grid(color='gray',alpha=0.6,linewidth=.5)
    ax.grid(color='gray',alpha=0.6,linewidth=.5)
    if interpolado:
        dir_bins = linspace(-dirdiff, 360 - dirdiff, dirrg)
        if (par in ['hs', 'tp', 'wind', 'energy']) and conv_oc:
            D -= 180           
        if np.any(D < 0):
            D[D < 0] += 360
        D[D > (360 - dirdiff)] -= 360
        P_bins_hist = linspace(0, Pmax, 10)
        # calcular o histograma 2d baseado nos bins
        table = (histogram2d(x=P,
                             y=D + dirdiff,
                             bins=[P_bins_hist, dir_bins],
                             normed=True))
        binarea = P_bins_hist[1] * dir_bins[1] * 200
        # grade para plot
        x, y = meshgrid(table[2][:], table[1][:])
        # calcular a porcentagem
        z = table[0] * binarea
        lowlev = z.max() * 0.05
        z[z < lowlev] = -lowlev * 0.9
        z = insert(z, z.shape[1], (z[:, 0]), axis=1)
        # Duplica ultima coluna para completar o diag polar
        z = insert(z, z.shape[0], (z[-1]), axis=0)
        # adiciona os zeros para que a interpolação não estoure o gráfico
        # converte para radianos
        x = radians(x)
        print(P_bins)
        # interpola todos os parametros
        z = ndimage.zoom(z, 5)
        x = ndimage.zoom(x, 5)
        y = ndimage.zoom(y, 5)
        # calcular a porcentagem
        # definir a prob max
        if MaxProb is None:
            MaxProb = z.max() * 1.1
        # plota dados,define os limites do grafico, palheta de cores, a origem
        cax = ax.contourf(x,
                          y,
                          z,
                          linspace(0, ceil(MaxProb), 200),
                          cmap=plt.cm.jet,
                          origin='lower',
                          antialiased=False)
        # numeros e a distância do grafico polar
        ff = ceil(MaxProb / 10) + 1
        cb = plt.colorbar(cax,
                          pad=.075,
                          shrink=0.8,
                          format='%i',
                          ticks=arange(0, ceil(MaxProb) + ff, ff))
        cb.ax.set_title(u'(%)\n')
        cb.mappable.set_clim(0, ceil(MaxProb))
        # define a cor em zero
        cb.vmin = 0
        # tipo
        tipo = u'_histogram_directional.png'
        # titulo
        titulo1 = u'Directional Histogram -'
    else:

        try:
            P = ma.masked_equal(P, 0)
            D = ma.masked_array(D, P.mask)
            P = ma.MaskedArray.compressed(P)
            D = ma.MaskedArray.compressed(D)
        except BaseException:
            pass

        dir_bins = linspace(-dirdiff, 360 - dirdiff, dirrg)
        if (par in ['hs', 'tp', 'wind', 'energy']) and conv_oc:
            D -= 180           
        if np.any(D < 0):
            D[D < 0] += 360
        if np.any(D > 360):
            D[D > 360] -= 360
        if np.any(D > (360 - dirdiff)):
            D[D > (360 - dirdiff)] -= 360
        # calcular o histograma 2d baseado nos bins
        table = (histogram2d(x=P,
                             y=D,
                             bins=[P_bins, dir_bins],
                             normed=False)
                 )
        theta = radians(table[2][:])[:-1]

        stat = cumsum(table[0], axis=0) / table[0].sum() * 100.
        legenda = []

        windcolors = flipud(
            plt.cm.hsv(
                list(map(
                    int, list(
                        round(
                            linspace(0, 180, len(P_bins)
                                     )
                        )
                    )
                )
                )
            )
        )
        for k in flipud(range(size(stat, 0))):
            ax.bar(theta + abs(np.min(theta)),
                   stat[k, :],
                   width=radians(dirdiff * 2),
                   bottom=0.0,
                   color=windcolors[k],
                   edgecolor="k")
            legenda.append('-'.join([str(table[1][k]), str(table[1][k + 1])]))
        if MaxProb is not None:
            ax.set_rmax(MaxProb)
        ax.tick_params(direction='out', length=8.5, color='r', zorder=10,labelsize=12,pad=-3)
        legenda[-1] = u'>' + str(table[1][k])
        if par == 'corrente':
            lg = plt.legend(legenda, bbox_to_anchor=(1.95, 1.15), title=cabecalho, prop={'size': 12})
            lg.get_title().set_fontsize(12)
        else:
            lg = plt.legend(legenda, bbox_to_anchor=(1.6, 1), title=cabecalho, prop={'size': 12})
            lg.get_title().set_fontsize(12)
        #plt.legend(loc='best')
        plt.tight_layout()
        tipo = u'_rose_directional.png'
        if par == 'wind':
            titulo1 = 'Wind Rose '
        else:
            titulo1 = 'Rose of '
    # adjust axis automatically
    if r_angle is None:
        r_angle = dir_bins[argmin(sum(table[0], axis=0))]
    ax.set_rlabel_position(r_angle)
    ax.set_title(titulo1 + titulo2, fontsize=12, y=1.15, x=0.85)
    if arqname == '':
        plt.show()
    else:
        # output file figure name
        plt.savefig(fname + tipo, format='png', dpi=300, bbox_inches='tight')
    # clear figure
    plt.clf()
    plt.cla()
    plt.close()
    print('Done')
     
    
    if arqname != '':
        D[D > 360] -= 360
        D[D > (360 - dirdiff)] -= 360
        dir_bins = linspace(0, 360, dirrg) - dirdiff
        # calculate the 2d histogram based on the bins
        table = (histogram2d(x=P,
                             y=D,
                             bins=[T_bins, dir_bins],
                             normed=porcentagem)
                 )
        if porcentagem:
            binarea = T_bins[1] * dir_bins[1] * 200
        else:
            binarea = 1
        # write output xlsx
        workbook = xlsxwriter.Workbook(fname + u'_conjoint_occurrence.xlsx')
        # gives the point name to the table
        worksheet = workbook.add_worksheet()
        # format information
        # column size in cm
        worksheet.set_column('A:B', 10)
        worksheet.set_column('B:S', 5)
        # create formats (already inserting bold)
        format1 = workbook.add_format({'bold': True,
                                       'font_name': 'Arial', 'font_size': 10,
                                       'align': 'center',
                                       'bg_color': '#B8CCE4'})
        format2 = workbook.add_format({'bold': False,
                                       'font_name': 'Arial', 'font_size': 10,
                                       'align': 'center',
                                       'bg_color': '#FFFFFF'})
        format3 = workbook.add_format({'bold': False,
                                       'font_name': 'Arial', 'font_size': 10,
                                       'align': 'center',
                                       'bg_color': '#FFFFFF'})
        format4 = workbook.add_format({'bold': False,
                                       'font_name': 'Arial', 'font_size': 10,
                                       'align': 'center',
                                       'bg_color': '#B8CCE4'})
        format5 = workbook.add_format({'bold': False,
                                       'font_name': 'Arial', 'font_size': 10,
                                       'align': 'center',
                                       'bg_color': '#B8CCE4'})
        format6 = workbook.add_format({'bold': True,
                                       'font_name': 'Arial', 'font_size': 10,
                                       'align': 'center',
                                       'bg_color': '#B8CCE4'})
        format7 = workbook.add_format({'bold': False,
                                       'font_name': 'Arial', 'font_size': 10,
                                       'align': 'center',
                                       'bg_color': '#B8CCE4'})
        # formatting of decimal places
        if porcentagem:
            format2.set_num_format('0.00')
        else:
            format2.set_num_format('0')
        format3.set_num_format('0.0')
        format5.set_num_format(myfmt)
        format7.set_num_format(myfmt)
        # insert cell division lines
        format4.set_top(1)
        format6.set_bottom(1)
        format7.set_bottom(1)

        # inserts the header into the file
        for k, hd in enumerate(head):
            worksheet.write(0, k, hd, format6)
        # write the occurrence lines
        for j in range((len(table[1]) - 1)):
            worksheet.write(
                j + 1,
                0,
                '-'.join([
                    str(table[1][j]),
                    str(table[1][j + 1])]
                ).replace('.', ','),
                format1)

            for i in range(len(head) - 2):
                worksheet.write(
                    j + 1, i + 1, table[0][j, i] * binarea, format2)
            if porcentagem:
                worksheet.write(j +
                                1, i +
                                2, np.sum(table[0][j, :]) *
                                binarea, format3)
            else:
                worksheet.write(
                    j + 1,
                    i + 2,
                    np.sum(table[0][j, :]) / np.sum(table[0]) * 100,
                    format3)
        # writes the total and the percentage of values per direction
        worksheet.write(j + 2, 0, u'(%)', format1)
        totais = np.sum(table[0], axis=0) * binarea
        if porcentagem is False:
            totais = totais / np.sum(totais) * 100
        for i, total in enumerate(totais):
            worksheet.write(j + 2, i + 1, total, format5)

        worksheet.write(j + 2, i + 3, '', format5)
        worksheet.write(j + 2, i + 2, '', format5)
        worksheet.write(j + 3, i + 3, '', format5)
        worksheet.write(j + 3, i + 2, '', format5)
        worksheet.write(j + 4, i + 3, '', format7)
        worksheet.write(j + 4, i + 2, '', format7)

        # writes the averages and maximums for each direction
        worksheet.write(j + 3, 0, u'Mean', format5)
        worksheet.write(j + 4, 0, u'Max.', format7)
        if np.ma.isMaskedArray(P) is False:
            P = np.ma.masked_object(P, -99999)
        for l in range(len(table[2]) - 1):
            if len(np.ma.MaskedArray.compressed(
                    P[(D > table[2][l]) & (D < table[2][l + 1])])) == 0:
                worksheet.write(j + 4, l + 1, 0, format7)
                worksheet.write(j + 3, l + 1, 0, format5)
            else:
                worksheet.write(
                    j + 4, l + 1, np.nanmax(
                        P[(D > table[2][l]) & (D < table[2][l + 1])]
                        ),
                    format7)
                worksheet.write(
                    j + 3, l + 1, np.nanmean(
                        P[(D > table[2][l]) & (D < table[2][l + 1])]
                        ),
                    format5)

        # closes the file
        workbook.close()
