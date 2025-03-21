from collections import Counter
from colorsys import hls_to_rgb
from copy import deepcopy
import json
import numpy as np

from fish_length import Fish_Length
from tracker_sort import Sort
from tracker_bytetrack import Associate
# import lib

class Tracker:
    def __init__(self, clip_info, algorithm=Sort, args={'max_age':1, 'min_hits':0, 'iou_threshold':0.05}, min_hits=3, reverse=False):
        self.algorithm = algorithm(**args)
        self.fish_ids = Counter()
        self.reverse = reverse
        self.min_hits = min_hits
        self.json_data = deepcopy(clip_info)
        if reverse:
            self.frame_id = self.json_data['end_frame']
        else:
            self.frame_id = self.json_data['start_frame']
        self.json_data['frames'] = []

    # Boxes should be given in normalized [x1,y1,x2,y2,c]
    def update(self, dets=np.empty((0, 5)),frame_num=None):
        if frame_num is not None:
            self.frame_id = frame_num
        new_frame_entries = []
        for track in self.algorithm.update(dets):

            # Match confidence with correct track
            conf = 0
            min_score = 1000000
            if type(self.algorithm) == Sort:
                for det in dets:
                    score = sum(abs(det[0:4] - track[0:4]))
                    if (score < min_score):
                        min_score = score
                        conf = det[4]
            elif type(self.algorithm) == Associate:
                for det in dets[0]:
                    score = sum(abs(det[0:4] - track[0:4]))
                    if (score < min_score):
                        min_score = score
                        conf = det[4]
                for det in dets[1]:
                    score = sum(abs(det[0:4] - track[0:4]))
                    if (score < min_score):
                        min_score = score
                        conf = det[4]
            
            # Assign Track
            self.fish_ids[int(track[4])] += 1
            new_frame_entries.append({
                'fish_id': int(track[4]),
                'bbox': list(track[:4]),
                'visible': 1,
                'human_labeled': 0,
                'conf': conf
            })
        new_frame_entries = sorted(new_frame_entries, key=lambda k: k['fish_id']) 

        self.json_data['frames'].append(
            {
                'frame_num': self.frame_id,
                'fish': new_frame_entries
            })
        if self.reverse:
            self.frame_id -= 1
        else:
            self.frame_id += 1
    
    def finalize(self, output_path=None, min_length=-1.0, min_travel=-1.0): # vert_margin=0.0
        json_data = deepcopy(self.json_data)
           
        # map (valid) fish IDs to 0, 1, 2, ...
        fish_id_map = {}
        for fish_id, count in self.fish_ids.items():
            if count >= self.min_hits:
                fish_id_map[fish_id] = len(fish_id_map)

        # separate frame boxes into tracks, keyed by mapped IDs
        # each track is a list of tuples ( bbox, frame_num )
        tracks = { v : [] for _, v in fish_id_map.items() }
        for frame in json_data['frames']:
            for bbox in frame['fish']:
                # check if valid
                if bbox['fish_id'] in fish_id_map.keys():
                    track_id = fish_id_map[bbox['fish_id']]
                    tracks[track_id].append((bbox['bbox'], frame['frame_num']))

        # map IDs and keep frame['fish'] sorted by ID
        for i, frame in enumerate(json_data['frames']):
            new_frame_entries = []
            for frame_entry in frame['fish']:
                if frame_entry['fish_id'] in fish_id_map:
                    frame_entry['fish_id'] = fish_id_map[frame_entry['fish_id']]
                    new_frame_entries.append(frame_entry)
            frame['fish'] = sorted(new_frame_entries, key=lambda k: k['fish_id'])

        # create summary 'fish' entry for json data
        json_data['fish'] = []
        for track_id, boxes in tracks.items():
            fish_entry = {}
            fish_entry['id'] = track_id
            fish_entry['length'] = -1
            
            # top = False
            # bottom = False
            # for frame in json_data['frames']:
            #     for frame_entry in frame['fish']:
            #         if frame_entry['fish_id'] == track_id:
            #             if frame_entry['bbox'][3] > vert_margin:
            #                 top = True
            #             if frame_entry['bbox'][1] < 1 - vert_margin:
            #                 bottom = True
            #             break

            # if not top or not bottom:
            #     continue
            
            start_bbox = boxes[0][0]
            end_bbox = boxes[-1][0]
            fish_entry['direction'] = Tracker.get_direction(start_bbox, end_bbox)

            fish_entry['travel_dist'] = Tracker.get_travel_distance(start_bbox, end_bbox, json_data['image_meter_width'], json_data['image_meter_height'])
            
            fish_entry['start_frame_index'] = boxes[0][1]
            fish_entry['end_frame_index'] = boxes[-1][1]
            fish_entry['color'] = Tracker.selectColor(track_id)

            json_data['fish'].append(fish_entry)


        # filter 'fish' field by fish length and travel distance
        json_data = Fish_Length.add_lengths(json_data)
        invalid_ids = []
        if min_length != -1.0:
            new_fish = []
            for fish in json_data['fish']:
                if fish['length'] > min_length and fish['travel_dist'] > min_travel:
                    new_fish.append(fish)
                else:
                    invalid_ids.append(fish['id'])
            json_data['fish'] = new_fish
        
        # filter 'frames' field by fish length
        if len(invalid_ids):
            for frame in json_data['frames']:
                new_fish = []
                for fish in frame['fish']:
                    if fish['fish_id'] not in invalid_ids:
                        new_fish.append(fish)
                frame['fish'] = new_fish
            
        if output_path is not None:
            with open(output_path,'w') as output:
                json.dump(json_data, output, indent=2)

        return json_data

    def state(self, output_path=None):
        json_data = deepcopy(self.json_data)

        if output_path is not None:
            with open(output_path,'w') as output:
                json.dump(json_data, output, indent=2)
        
        return json_data

    @staticmethod
    def selectColor(number):
        hue = ((number * 137.508 + 60) % 360) / 360
        return '#{0:02x}{1:02x}{2:02x}'.format(*(int(n * 255) for n in hls_to_rgb(hue, 0.5, 0.75)))

    @staticmethod
    def get_direction(start_bbox, end_bbox):
        start_center = (start_bbox[2] + start_bbox[0])/2
        end_center = (end_bbox[2] + end_bbox[0])/2
        if start_center < 0.5 and end_center >= 0.5:
            return 'right'
        elif start_center >= 0.5 and end_center < 0.5:
            return 'left'
        else:
            return 'none'

    @staticmethod
    def get_travel_distance(start_bbox, end_bbox, image_meter_width, image_meter_height):
        dx = (start_bbox[2] + start_bbox[0])/2 - (end_bbox[2] + end_bbox[0])/2
        dx *= image_meter_width
        dy = (start_bbox[3] + start_bbox[1])/2 - (end_bbox[3] + end_bbox[1])/2
        dy *= image_meter_height
        return np.sqrt(dx*dx + dy*dy)

    @staticmethod
    def count_dirs(json_data):
        right = 0
        left = 0
        none = 0
        for fish_entry in json_data['fish']:
            if fish_entry['direction'] == 'right':
                right += 1
            elif fish_entry['direction'] == 'left':
                left += 1
            else:
                none += 1
        return (right, left, none)