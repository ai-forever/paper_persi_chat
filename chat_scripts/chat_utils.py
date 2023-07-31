import portion as P


def check_text_coverage(cur_interval, covered_intervals, thresh=0.8):
    '''
    Checks already covered spans:  interval_len / intersection_len > thresh => covered
    '''
    sum_intersection_len = 0
    for covered_interval in covered_intervals:
        intersection = P.to_data(cur_interval & covered_interval)
        sum_intersection_len += intersection[0][2] - intersection[0][1] if len(intersection) > 0 else 0
    cur_interval_python = P.to_data(cur_interval)
    cur_interval_len = cur_interval_python[0][2] - cur_interval_python[0][1] if len(cur_interval_python) > 0 else 0
    if cur_interval_len == 0 or sum_intersection_len / cur_interval_len > thresh:
        return True
    return False


def join_segments(raw_segments, max_len=4500):
    '''
    Join paper segements (by same class and limited by max_len)
    '''
    
    indeces_sections = [[0,1]]
    
    prev_title = raw_segments[2][-1]
    i = 2
    cur_section = []
    while i < len(raw_segments):
        cur_title = raw_segments[i][-1]
        if cur_title == prev_title:
            cur_section.append(i)
        else:
            indeces_sections.append(cur_section)
            cur_section = [i]
            prev_title = cur_title
        i += 1
    
    if len(cur_section) > 0:
        indeces_sections.append(cur_section)
    
    joined_segments = []
    for sec in indeces_sections:
        cur_text = ''
        cur_split = []
        
        for idx in sec:
            if len(cur_text + '\n' + raw_segments[idx][1]) < max_len or len(cur_text) == 0:
                cur_split.append({'id': raw_segments[idx][0],
                                  'title': raw_segments[idx][-2], 'section_type': raw_segments[idx][-1]})
                cur_text = cur_text + '\n' + raw_segments[idx][1]
                cur_text = cur_text.strip()
            
            else:
                joined_segments.append((cur_text, cur_split))
                cur_text = raw_segments[idx][1]
                cur_split = [{'id': raw_segments[idx][0],
                              'title': raw_segments[idx][-2], 'section_type': raw_segments[idx][-1]}]
        
        if len(cur_text) > 0:
            joined_segments.append((cur_text, cur_split))
                
    return joined_segments
