"""Current development task exposes one configured world-coordinate target."""


def visual_target_slots(config):
    slots=config['conveyor']['slots_xy']
    index=config.get('visual_task',{}).get('target_slot_index',0)
    if type(index) is not int or not 0<=index<len(slots):
        raise ValueError('Visual task requires one valid target_slot_index')
    return (tuple(slots[index]),)
