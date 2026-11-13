"""Single-carton target and visually observed conveyor slots for the paired task."""


def visual_target_slots(config):
    slots=config['conveyor']['slots_xy']
    index=config.get('visual_task',{}).get('target_slot_index',0)
    if type(index) is not int or not 0<=index<len(slots):
        raise ValueError('Visual task requires one valid target_slot_index')
    return (tuple(slots[index]),)


class ConveyorSlots:
    """The paired controller owns reservations; perception owns availability."""

    def __init__(self, config):
        self.positions = tuple(tuple(xy) for xy in config['conveyor']['slots_xy'])
        if len(set(self.positions)) != len(self.positions):
            raise ValueError('Conveyor slot centers must be distinct')
        self.states = {xy: 'unknown' for xy in self.positions}
        self.owners = {}
        self.observed_tracks = {}
        self.timestamps = {}

    def observe(self, workspace):
        for xy in self.positions:
            view = workspace.slot(xy)
            if view['observation_time_s'] <= self.timestamps.get(xy, float('-inf')):
                continue
            self.timestamps[xy] = view['observation_time_s']
            self.observed_tracks[xy] = tuple(view['observed_objects'])
            if view['free']:
                self.states[xy] = 'reserved' if xy in self.owners else 'free'
            else:
                self.states[xy] = 'occupied' if view['observed_objects'] else 'unknown'

    def reserve_pair(self, assignments):
        # Validate the entire assignment before committing either reservation.
        if len(assignments) != 2 or len(set(assignments.values())) != 2:
            raise ValueError('Each arm needs a distinct slot')
        if any(xy in self.owners or arm in self.owners.values() or self.states.get(xy) != 'free'
               for arm, xy in assignments.items()):
            raise RuntimeError('PAIR_SLOT_UNAVAILABLE')
        for arm, xy in assignments.items():
            self.owners[xy] = arm
            self.states[xy] = 'reserved'

    def confirm_placement(self, arm, xy, observed_track):
        if (self.owners.get(xy) != arm or self.states[xy] != 'occupied'
                or not observed_track or self.observed_tracks[xy] != (observed_track,)):
            raise RuntimeError('PLACEMENT_NOT_CONFIRMED')
        del self.owners[xy]

    def report(self):
        return [dict(xy=xy, state=self.states[xy], owner=self.owners.get(xy),
                     observation_time_s=self.timestamps.get(xy)) for xy in self.positions]
