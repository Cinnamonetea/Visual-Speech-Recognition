from moviepy import *
import numpy as np
import pandas as pd

def time_to_sec(time_str):
    time_str = str(time_str).strip()
    parts = time_str.split(':')
    minutes = int(parts[0])
    seconds = float(parts[1])
    return int(minutes * 60 + seconds)

def sec_to_time(sec):
    minutes = int(sec // 60)
    seconds = sec % 60
    return f"{minutes}:{seconds:02.0f}"

def cut_video_and_transcript(video_path, cuts_path, transcript_path,
                               output_video_path, output_transcript_path):
    # convert sections time into seconds
    df_cuts = pd.read_csv(cuts_path)
    cuts = []
    for _, row in df_cuts.iterrows():
        start = time_to_sec(str(row['start']))
        end = time_to_sec(str(row['end']))
        cuts.append((start, end))

    video = VideoFileClip(video_path)

    # find necessary parts
    keep_parts = []
    current_time = 0

    for start, end in cuts:
        if current_time < start:
            keep_parts.append((current_time, start))
        current_time = end
    if current_time < video.duration:
            keep_parts.append((current_time, video.duration))

    # cutout necessary clips
    keep_clips = []
    for start, end in keep_parts:
        clip = video.subclipped(start, end)
        keep_clips.append(clip)

    edited_video = concatenate_videoclips(keep_clips)
    edited_video.write_videofile(output_video_path)

    video.close()
    edited_video.close()

    df_transcript = pd.read_csv(transcript_path)

    # convert time into seconds
    df_transcript['time_sec'] = df_transcript['time'].apply(time_to_sec)

    def is_in_cut(time_sec):
        for start, end in cuts:
            if start <= time_sec <= end:
                return True
        return False

    filtered_df = df_transcript[~df_transcript['time_sec'].apply(is_in_cut)].copy()


    def adjust_time(time_sec):
        removed_duration = 0
        for start, end in cuts:
            if time_sec > start:
                removed_duration += min(end, time_sec) - start
            else:
                break
        return time_sec - removed_duration

    filtered_df['adjusted_time_sec'] = filtered_df['time_sec'].apply(adjust_time)
    filtered_df['time'] = filtered_df['adjusted_time_sec'].apply(sec_to_time)
    filtered_df[['time', 'text']].to_csv(output_transcript_path, index=False)

# cut_video_and_transcript("./video/original/part15.mp4", './csv_format/trash_pt15.csv', './csv_format/part15_text.csv',
#                                "./dataset/video/pt15.mp4", './dataset/transcript/pt15.csv')
