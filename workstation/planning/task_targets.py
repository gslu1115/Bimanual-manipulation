"""Single-carton target and visually observed conveyor slots for the paired task."""


def visual_target_slots(config):
    slots=config['conveyor']['slots_xy']
    index=config.get('visual_task',{}).get('target_slot_index',0)
    if type(index) is not int or not 0<=index<len(slots):
        raise ValueError('Visual task requires one valid target_slot_index')
    return (tuple(slots[index]),)


def reobserve_reserved_slot(slots,arm,xy,after,observe,candidates,try_adjust,checks):
    """Require new empty depth; only checked, bounded viewpoint moves may retry.

    try_adjust returns False for a planning rejection only. Execution and
    payload-observation errors propagate; neither can trigger another move.
    """
    if slots.owners.get(xy)!=arm:raise RuntimeError('SLOT_REOBSERVE_RESERVATION_LOST')
    candidates=iter(candidates)
    while True:
        view=observe();checks.append(dict(kind='observation',view=dict(view)))
        time=view['observation_time_s']
        if tuple(view['xy'])!=xy or time<=after:
            raise RuntimeError('SLOT_REOBSERVE_STALE_OR_WRONG_VIEW')
        after=time
        if slots.owners.get(xy)!=arm:raise RuntimeError('SLOT_REOBSERVE_RESERVATION_LOST')
        if view['free'] and not view['observed_objects'] and view['reason'] is None:return view
        if view['observed_objects'] or view['reason']!='UNOBSERVED_PATH_SPACE':
            raise RuntimeError('SLOT_REOBSERVE_NOT_EMPTY:'+str(view['reason']))
        for candidate in candidates:
            if try_adjust(candidate):break
        else:raise RuntimeError('SLOT_REOBSERVE_NO_VISIBLE_CHECKED_POSE')


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
        self.apply_views((xy,workspace.slot(xy)) for xy in self.positions)

    def apply_views(self, views):
        for xy,view in views:
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
        if not self.placement_observed(arm, xy, observed_track):
            raise RuntimeError('PLACEMENT_NOT_CONFIRMED')
        del self.owners[xy]

    def placement_observed(self, arm, xy, observed_track):
        return (self.owners.get(xy) == arm and self.states.get(xy) == 'occupied'
                and bool(observed_track) and self.observed_tracks.get(xy) == (observed_track,))

    def approach_clearance(self,arm,xy,current,prior,*,accounted_for,stationary_belt):
        """Reuse a recent free view only while all cartons are accounted for on a stopped belt."""
        if self.owners.get(xy)!=arm:return None
        if current['free']:return 'current_depth'
        if (stationary_belt and accounted_for and prior is not None and prior['free']
                and tuple(prior['xy'])==xy and tuple(current['xy'])==xy
                and 0<current['observation_time_s']-prior['observation_time_s']<=20.
                and current['reason']=='UNOBSERVED_PATH_SPACE' and not current['observed_objects']):
            return 'observed_empty_before_transfer_both_cartons_in_hand'
        return None

    def report(self):
        return [dict(xy=xy, state=self.states[xy], owner=self.owners.get(xy),
                     observation_time_s=self.timestamps.get(xy)) for xy in self.positions]
