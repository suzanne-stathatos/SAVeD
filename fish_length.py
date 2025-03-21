from collections import defaultdict
from copy import deepcopy
import json
import numpy as np

class Fish_Length:
    @staticmethod
    def mean_length(tracks, constant, aux=-1):
        return [np.mean(track[2] - track[0])*constant for track in tracks]

    @staticmethod
    def quantile_length(tracks, constant, aux=-1):
        return [np.quantile(track[2] - track[0], aux)*constant for track in tracks]

    @staticmethod
    def quantile_diagonal(tracks, constant, aux=-1):
        return [np.quantile(np.sqrt((track[2] - track[0])**2 + (track[3] - track[1])**2), aux)*constant for track in tracks]

    @staticmethod
    def add_lengths(json_data, length_fn=quantile_length.__func__, constant=0.8348286633599985, aux=0.8773333335319834, output_path=None):
        json_data = deepcopy(json_data)
        
        tracks = defaultdict(list)
        for frame in json_data['frames']:
            for frame_entry in frame['fish']:
                tracks[frame_entry['fish_id']].append(np.array(frame_entry['bbox']))
        tracks = [np.array(track).T for _, track in sorted(tracks.items())]

        lengths = np.array(length_fn(tracks, constant*json_data['image_meter_width'], aux=aux))
        
        for fish, fish_length in zip(sorted(json_data['fish'], key=lambda k: k['id']), lengths):
            fish['length'] = fish_length
        
        if output_path is not None:
            with open(output_path,'w') as output:
                json.dump(json_data, output, indent=2)

        return json_data