%load_ext autoreload
%autoreload 2

import random

import librosa
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
import numpy as np
import scienceplots
import soundfile as sf

plt.style.use(['science'])

from src.utils import (
    load_json, load_pitch_track, write_pitch_track, cpath, load_annotations, get_plot_kwargs, 
    pitch_seq_to_cents, subsample_series, interpolate_below_length, smooth_pitch_curve,
    align_time_series, write_pkl, expand_zero_regions)


interp = 100
smoothing_factor = 0.4
subsample = 0.5
plot = False
plot_sample = None
out_path = 'data/bhairavi.pkl'

############
## Load Data
############
metadata_path = 'data/metadata.json'
data = load_json(metadata_path)

sr = 44100

print('Loading data...')
for track, d in data.items():
    print(f'    {track}')
    pitch_track = load_pitch_track(d['pitch_path'])

    d['pitch_raw'] = pitch_track[:,1]
    d['time_raw'] = pitch_track[:,0]
    d['timestep_raw'] = d['time_raw'][3]-d['time_raw'][2]

    d['annotations'] = load_annotations(d['annotations_path'])

    d['vocal'], _ = librosa.load(d['audio_path'], sr=sr)

    d['plot_kwargs_cents'] = get_plot_kwargs(d['raga'], d['tonic'], cents=True)
    d['plot_kwargs_hertz'] = get_plot_kwargs(d['raga'], d['tonic'], cents=False)




##########
## Process
##########
print("Processing pitch curves...")
for track, d in data.items():
    print(f'    {track}')
    pitch_raw = d['pitch_raw']
    time_raw = d['time_raw']
    timestep_raw = d['timestep_raw']
    beat_length = d['beat_length']
    tonic = d['tonic']

    annotations = d['annotations']

    # dont interpolate at boundaries
    boundaries = np.concatenate([annotations['start_sec'].values, annotations['end_sec'].values])
    boundaries_seq = [round(b/timestep_raw) for b in boundaries]


    pitch = expand_zero_regions(pitch_raw, round(0.02/timestep_raw))

    # Interpolation
    pitch = interpolate_below_length(pitch, 0, (interp*0.001/timestep_raw), boundaries_seq)

    null_ind = pitch==0

    pitch[pitch<50]=0
    pitch[null_ind]=0

    # cents
    pitch = pitch_seq_to_cents(pitch, tonic=tonic)
    
    # subsample
    time, pitch = subsample_series(time_raw, pitch, subsample)
    
    # beats
    time_beat = time/beat_length

    # smoothing
    pitch = smooth_pitch_curve(time_beat, pitch, smoothing_factor=smoothing_factor)
    
    d['pitch'] = pitch
    d['time'] = time_beat
    d['timestep'] = time_beat[3] - time_beat[2]


# Ensure equal timesteps
print("Aligning pitch curves...")

# Example usage:
# time_series1 = ([pitch data1], [time data1])
# time_series2 = ([pitch data2], [time data2])
# time_series3 = ([pitch data3], [time data3])
# interpolated_pitches, common_time_grid = interpolate_multiple_time_series(1000, time_series1, time_series2, time_series3)

pitch_time = []
for track, d in data.items():
    pitch_time.append((d['pitch'], d['time']))

result = align_time_series(*pitch_time)

for i, d in enumerate(data.values()):
    d['pitch'] = result[i][0]
    d['time'] = result[i][1]
    d['timestep'] = d['time'][1]-d['time'][0]



#######
## Plot
#######
def format_func(value, tick_number):
    minutes = int(value // 60)
    seconds = int(value % 60)
    milliseconds = int((value % 1) * 1000)
    return f'{minutes}:{seconds:02d}.{milliseconds:03d}'


titlesize = 6
ticksize = 5
labelsize = 5
gridweight = 0.3
annot_lineweight = 0.5
figsize=(3,1.4)
color='darkorange'
contextcolor='lightslategrey'
rec_color='gainsboro'  

skipped = {t:[] for t in data}

if plot:
    print('Plotting...')
    for track, d in data.items():
        
        if track == 'kamakshi':
            continue

        print(f'    {track}')
        annotations = d['annotations']
        pitch = d['pitch']
        time = d['time']
        timestep = d['timestep']
        beat_length = d['beat_length']
        plot_kwargs = d['plot_kwargs_cents']
        tonic = d['tonic']
        vocal = d['vocal']

        #annotations = annotations[~annotations['label'].str.contains('\?')]

        indices = annotations.index
        count = 0
        for I in range(len(indices)-2):
            
            curr_row = annotations.loc[indices[I]]
            prev_row = annotations.loc[indices[I-1]]
            next_row = annotations.loc[indices[I+1]]
            
            curri = curr_row.name

            prev_text_x = prev_row['start_sec']+(prev_row['end_sec']-prev_row['start_sec'])/2
            curr_text_x = curr_row['start_sec']+(curr_row['end_sec']-curr_row['start_sec'])/2
            next_text_x = next_row['start_sec']+(next_row['end_sec']-next_row['start_sec'])/2

            L = curr_row['label']
            LS = format_func(curr_row['start_sec'],1)
            
            if curr_row['start_sec']-prev_row['end_sec']>0.025:
                skipped[track].append(i)
                #continue

            if next_row['start_sec']-curr_row['end_sec']>0.025:
                skipped[track].append(i)
                #continue

            title = f'{L} at {LS}'




            # prev row
            start = prev_row['start_sec']
            end = prev_row['end_sec']

            s1_beats = start/beat_length
            s2_beats = end/beat_length

            s1 = round(s1_beats/timestep)
            s2 = round(s2_beats/timestep)

            prev_psamp = pitch[s1:s2]
            prev_tsamp_beats = time[s1:s2]
            prev_tsamp = prev_tsamp_beats*beat_length

            if sum(prev_psamp == None) > len(prev_psamp)/4:
                skipped[track].append(i)
                #continue




            # curr row
            start = curr_row['start_sec']
            end = curr_row['end_sec']

            s1_beats = start/beat_length
            s2_beats = end/beat_length

            s1 = round(s1_beats/timestep)
            s2 = round(s2_beats/timestep)

            curr_psamp = pitch[s1:s2]
            curr_tsamp_beats = time[s1:s2]
            curr_tsamp = curr_tsamp_beats*beat_length
        
            if sum(curr_psamp == None) > len(curr_psamp)/4:
                skipped[track].append(i)
                #continue




            # next row
            start = next_row['start_sec']
            end = next_row['end_sec']

            s1_beats = start/beat_length
            s2_beats = end/beat_length

            s1 = round(s1_beats/timestep)
            s2 = round(s2_beats/timestep)

            next_psamp = pitch[s1:s2]
            next_tsamp_beats = time[s1:s2]
            next_tsamp = next_tsamp_beats*beat_length
            
            if sum(next_psamp == None) > len(next_psamp)/4:
                skipped[track].append(i)
                #continue

            all_samp = np.concatenate([curr_psamp,prev_psamp,next_psamp])

            minp = np.nanmin(all_samp[all_samp != None])-200
            maxp = np.nanmax(all_samp[all_samp != None])+200

            yticks_dict = {k:v for k,v in plot_kwargs['yticks_dict'].items() if v >= minp and v <= maxp}
            
            plot_path = cpath(f'plots/svaras/{track}/{L}/{curri}_{L}.png')
            vpath = cpath(f'plots/svaras/{track}/{L}/{curri}_{L}.wav')

            # Plot
            print(f'      {plot_path}')
            plt.close('all')
            fig, ax = plt.subplots(1)
            fig.set_size_inches(figsize[0], figsize[1])
            
            rect = Rectangle((curr_row['start_sec'], minp), curr_row['end_sec']-curr_row['start_sec'], maxp-minp, facecolor=rec_color)
            ax.add_patch(rect)
            

            ax.plot(prev_tsamp, prev_psamp, color=contextcolor)
            ax.plot(curr_tsamp, curr_psamp, color=color)
            ax.plot(next_tsamp, next_psamp, color=contextcolor)
            ax.set_title(title, fontsize=titlesize)

            ax.set_ylabel(f'Cents as svara \npositions', fontsize=labelsize)
            tick_names = list(yticks_dict.keys())
            tick_loc = list(yticks_dict.values())
            ax.set_yticks(tick_loc)
            ax.set_yticklabels(tick_names)
            ax.grid(zorder=8, linestyle='--', linewidth=gridweight)
            
            if not np.isnan(maxp) and not np.isnan(minp):
                ax.set_ylim((minp, maxp))
            
            ax.set_xlim((prev_row['start_sec'], next_row['end_sec']))

            ax.set_title(title, fontsize=titlesize)

            ax.set_xlabel('Time (s)', fontsize=labelsize)

            ## Add text
            text_y = minp + (maxp - minp)*0.8

            plt.text(prev_text_x, text_y, prev_row['label'])
            plt.text(curr_text_x, text_y, curr_row['label'])
            plt.text(next_text_x, text_y, next_row['label'])
            
            plt.axvline(curr_tsamp[0],linestyle='--', lw=0.5, color='black')
            plt.axvline(curr_tsamp[-1],linestyle='--', lw=0.5, color='black')

            ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))

            plt.xticks(fontsize=ticksize, zorder=2)
            plt.yticks(fontsize=ticksize, zorder=2)

            plt.tight_layout()
            plt.savefig(plot_path, dpi=500)
            plt.close('all')

            if plot_sample:
                count += 1
                if count > plot_sample:
                    break

            s1 = round(curr_row['start_sec']*sr)
            s2 = round(curr_row['end_sec']*sr)
            
            vsamp = vocal[s1:s2]

            sf.write(vpath, vsamp, sr)




########
## Write
########
write_pkl(data, out_path)