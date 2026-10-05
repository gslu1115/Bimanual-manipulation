"""Simulator-independent episode contracts. SI units; quaternions are wxyz.

Ground truth supports evaluation and the explicitly labelled single-box control
baseline. The future vision-based pipeline must supply its own grasp candidates;
passing this baseline does not validate perception or a learned policy.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass


def rotation(q):
    n = math.sqrt(sum(v * v for v in q))
    if n < 1e-12:
        raise ValueError("Invalid zero quaternion")
    w, x, y, z = [v / n for v in q]
    return ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
            (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
            (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))


def orientation_error(q):
    r = rotation(q)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, r[2][2]))))
    yaw = math.atan2(r[1][0], r[0][0])
    # A tape line defines an axis: 0 and 180 degrees are equivalent.
    axis_yaw = abs((yaw + math.pi / 2) % math.pi - math.pi / 2)
    return tilt, math.degrees(axis_yaw)


def norm(v):
    return math.sqrt(sum(x*x for x in v))


@dataclass
class BoxState:
    name: str
    position: list
    orientation_wxyz: list
    linear_velocity: list
    angular_velocity: list

    def to_dict(self):
        return asdict(self)


def release_states(c, seed):
    """Non-intersecting release poses, not manually designed settled poses."""
    rng = random.Random(seed)
    u, b = c['upstream'], c['boxes']
    # Pack release points into low layers instead of a tall column. Bounding
    # spheres never intersect, including randomized orientation and small jitter.
    pitch = norm(b['size']) + .016
    cols, rows = [max(1, math.floor(v/pitch)+1) for v in u['release_size_xy']]
    cells = [((ix-(cols-1)/2)*pitch, (iy-(rows-1)/2)*pitch)
             for ix in range(cols) for iy in range(rows)]
    jitter = [min(.003, max(0., (u['release_size_xy'][j]-(n-1)*pitch)/2))
              for j, n in enumerate((cols, rows))]
    rng.shuffle(cells)
    result = []
    for i in range(b['count']):
        # Uniform random rotations (Shoemake), scalar-first.
        a, v, t = rng.random(), rng.random()*2*math.pi, rng.random()*2*math.pi
        q = [math.sqrt(a)*math.cos(t), math.sqrt(1-a)*math.sin(v),
             math.sqrt(1-a)*math.cos(v), math.sqrt(a)*math.sin(t)]
        cell = cells[i % len(cells)]
        p = [u['center_xy'][j] + cell[j] + rng.uniform(-jitter[j], jitter[j]) for j in range(2)]
        p.append(c['table']['top_z'] + u['release_clearance'] + (i//len(cells))*u['layer_gap'])
        velocity = [rng.uniform(-u['horizontal_speed'], u['horizontal_speed']) for _ in range(2)] + [0.0]
        result.append(BoxState(f'box_{i:02d}', p, q, velocity, [0., 0., 0.]))
    return result


def validate(c):
    def finite_tree(obj):
        if isinstance(obj, dict):
            for v in obj.values(): finite_tree(v)
        elif isinstance(obj, list):
            for v in obj: finite_tree(v)
        elif isinstance(obj, (int, float)) and not math.isfinite(obj):
            raise ValueError('All numbers must be finite')
    finite_tree(c)
    if c['schema_version'] != 2: raise ValueError('Expected schema version 2')
    b, u, belt = c['boxes'], c['upstream'], c['conveyor']
    if type(b['count']) is not int or not 1 <= b['count'] <= 18:
        raise ValueError('Box count must be 1..18')
    for dims in (b['size'], c['table']['size'], belt['size'], belt['slot_size_xy'], u['release_size_xy']):
        if not all(v > 0 for v in dims): raise ValueError('Dimensions must be positive')
    if b['mass_kg'] <= 0 or not 0 < c['physics']['dt'] <= 1/60:
        raise ValueError('Invalid mass or physics dt')
    if u['layer_gap'] <= norm(b['size']) or u['release_clearance'] <= norm(b['size'])/2:
        raise ValueError('Release layers must not intersect, for any orientation')
    if len(belt['slots_xy']) != 3 or len(c['robots']) != 2:
        raise ValueError('This task requires three slots and two arms')
    if b['size'][1] >= .08: raise ValueError('Default Panda gripper requires box width < 80 mm')
    for xy in belt['slots_xy']:
        if any(abs(xy[j]-belt['center_xy'][j]) + belt['slot_size_xy'][j]/2 > belt['size'][j]/2 for j in range(2)):
            raise ValueError('Slot outside belt')
    for i, xy in enumerate(belt['slots_xy']):
        for other in belt['slots_xy'][i+1:]:
            if all(abs(xy[j]-other[j]) < belt['slot_size_xy'][j] for j in range(2)):
                raise ValueError('Slots overlap')
    if belt['speed'] <= 0 or belt['index_distance'] <= 0:
        raise ValueError('Conveyor travel and speed must be positive')
    p = c['physics']
    if not 0 <= p['dynamic_friction'] <= p['static_friction'] or not 0 <= p['restitution'] <= 1:
        raise ValueError('Invalid contact material')
    for cam in c['cameras']:
        if len(cam['resolution']) != 2 or any(type(v) is not int or v < 16 for v in cam['resolution']):
            raise ValueError('Invalid camera resolution [width, height]')
    if 'single_pick_place' in c:
        s = c['single_pick_place']
        if not isinstance(s, dict):
            raise ValueError('single_pick_place must be an object')
        required = ('arm', 'box_xy', 'box_yaw_deg', 'slot_index', 'lift_height',
                    'move_speed', 'joint_speed', 'grip_position', 'grip_max_force', 'verify_timeout_s')
        if any(k not in s for k in required):
            raise ValueError('Incomplete single_pick_place configuration')
        if s['arm'] not in [arm['name'] for arm in c['robots']]:
            raise ValueError('Single-box arm must name a configured robot')
        if type(s['slot_index']) is not int or not 0 <= s['slot_index'] < len(belt['slots_xy']):
            raise ValueError('Single-box slot_index must be an integer from 0 to 2')
        xy = s['box_xy']
        if not isinstance(xy, list) or len(xy) != 2:
            raise ValueError('Single-box box_xy must be [x, y]')
        numbers = list(xy) + [s[k] for k in required if k not in ('arm', 'box_xy', 'slot_index')]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in numbers):
            raise ValueError('Single-box parameters must be finite numbers')
        if any(s[k] <= 0 for k in ('lift_height', 'move_speed', 'joint_speed', 'grip_max_force', 'verify_timeout_s')):
            raise ValueError('Single-box lift, speeds, force and timeout must be positive')
        if not 0 <= s['grip_position'] < min(.04, b['size'][1]/2):
            raise ValueError('Single-box grip_position must close below half the box width within Panda finger limits')
        if s['verify_timeout_s'] < c['verification']['hold_s']:
            raise ValueError('Single-box verification timeout must cover the required hold time')
        yaw = math.radians(s['box_yaw_deg'])
        cs, sn = abs(math.cos(yaw)), abs(math.sin(yaw))
        extents = [(cs*b['size'][0]+sn*b['size'][1])/2,
                   (sn*b['size'][0]+cs*b['size'][1])/2]
        # The tray rims are 12 mm thick and centered on the nominal boundary.
        if any(abs(xy[j]-u['center_xy'][j])+extents[j] >= u['tray_size_xy'][j]/2-.006 for j in range(2)):
            raise ValueError('Single-box footprint must lie strictly inside the tray rims')
    if 'dual_handover' in c:
        h = c['dual_handover']
        names = [r['name'] for r in c['robots']]
        if h['giver'] not in names or h['receiver'] not in names or h['giver'] == h['receiver']:
            raise ValueError('Handover needs two distinct configured arms')
        if type(h['slot_index']) is not int or not 0 <= h['slot_index'] < 3:
            raise ValueError('Invalid handover slot')
        if len(h['box_xy']) != 2 or len(h['handover_center']) != 3:
            raise ValueError('Invalid handover positions')
        if any(type(v) not in (int, float) or not math.isfinite(v)
               for v in h['box_xy']+h['handover_center']):
            raise ValueError('Handover positions must be finite numbers')
        for k in ('lift_height', 'approach_distance', 'move_speed', 'angular_speed', 'joint_speed',
                  'grip_max_force', 'slip_position_m', 'slip_angle_deg', 'verify_timeout_s'):
            if type(h[k]) not in (int, float) or not math.isfinite(h[k]) or h[k] <= 0:
                raise ValueError('Handover speeds, tolerances, force and duration must be positive')
        if not 0 <= h['grip_position'] < b['size'][1]/2 < .04:
            raise ValueError('Handover finger target must grip the carton within Panda limits')
        if not -b['size'][0]/2+.01 < h['giver_offset_x'] < 0 < h['receiver_offset_x'] < b['size'][0]/2-.01:
            raise ValueError('Handover grips must lie on opposite longitudinal halves')
        if h['handover_center'][2] < c['table']['top_z']+.20:
            raise ValueError('Insufficient handover clearance above the table')
        if h['verify_timeout_s'] < c['verification']['hold_s']:
            raise ValueError('Handover timeout must cover verification dwell')
        if any(abs(h['box_xy'][j]-u['center_xy'][j])+b['size'][j]/2 >= u['tray_size_xy'][j]/2-.006 for j in range(2)):
            raise ValueError('Inverted carton must start inside the tray')
    return c


def slot_errors(box, xy, c):
    tilt, yaw = orientation_error(box.orientation_wxyz)
    expected_z = c['conveyor']['top_z'] + c['boxes']['size'][2]/2
    return {'xy_m': math.hypot(box.position[0]-xy[0], box.position[1]-xy[1]),
            'z_m': abs(box.position[2]-expected_z), 'tilt_deg': tilt, 'axis_yaw_deg': yaw,
            'linear_speed': norm(box.linear_velocity), 'angular_speed': norm(box.angular_velocity)}


def slot_valid(errors, c):
    v = c['verification']
    return all(errors[k] <= v[t] for k, t in (
        ('xy_m', 'xy_tolerance'), ('z_m', 'z_tolerance'), ('tilt_deg', 'tilt_deg'),
        ('axis_yaw_deg', 'yaw_deg'), ('linear_speed', 'linear_speed'), ('angular_speed', 'angular_speed')))


def single_pnp_success(errors, c, lift_m, stable_s, finger_positions, arm_home):
    """Evaluate measured release/retreat evidence in addition to the target pose.

    The simulator supplies maximum measured lift above the initial box center,
    uninterrupted valid-slot dwell after release, and current finger positions.
    It resets the dwell whenever slot_valid fails; elapsed action time is not
    valid dwell. Merely starting a box at the target cannot pass this contract.
    """
    if not arm_home or len(finger_positions) != 2:
        return False
    numbers = [lift_m, stable_s, *finger_positions]
    if not all(math.isfinite(v) for v in numbers):
        return False
    return (lift_m >= .08 and stable_s >= c['verification']['hold_s']
            and all(v > .037 for v in finger_positions) and slot_valid(errors, c))


class IndexingConveyor:
    """WAITING -> INDEXING -> WAITING; faults require explicit recovery.

    This class returns belt velocity; the simulator applies physical surface
    velocity. Box poses are never overwritten to simulate successful transport.
    """
    def __init__(self, c):
        self.c = c
        self.state = 'WAITING'
        self.occupied = [None]*3
        self.candidate = [None]*3
        self.dwell = [0.]*3
        self.batch = []
        self.delivered = set()
        self.elapsed = 0.
        self.distance = 0.
        self.completed_batches = 0
        self.events = []

    def update(self, boxes, dt, safe_to_index=True):
        if self.state == 'FAULT': return 0.
        by_name = {b.name: b for b in boxes}
        if self.state == 'INDEXING':
            if not safe_to_index:
                self.state = 'FAULT'
                self.events.append({'event': 'interlock_open_during_index'})
                return 0.
            self.elapsed += dt
            self.distance += self.c['conveyor']['speed']*dt
            cleared = all(n in by_name and by_name[n].position[0] -
                          sum(abs(rotation(by_name[n].orientation_wxyz)[0][j])*self.c['boxes']['size'][j]/2 for j in range(3))
                          > self.c['conveyor']['clearance_x'] and
                          abs(by_name[n].position[1]-self.c['conveyor']['center_xy'][1]) < self.c['conveyor']['size'][1]/2 and
                          by_name[n].position[2] > self.c['conveyor']['top_z'] for n in self.batch)
            if self.distance >= self.c['conveyor']['index_distance'] and cleared:
                self.delivered.update(self.batch)
                self.completed_batches += 1
                self.events.append({'event': 'batch_cleared', 'boxes': list(self.batch)})
                self.state = 'WAITING'
                self.occupied, self.candidate, self.dwell = [None]*3, [None]*3, [0.]*3
                return 0.
            if self.elapsed > self.c['conveyor']['timeout_s']:
                self.state = 'FAULT'
                self.events.append({'event': 'index_timeout', 'boxes': list(self.batch)})
                return 0.
            return self.c['conveyor']['speed']
        for i, xy in enumerate(self.c['conveyor']['slots_xy']):
            valid = [b.name for b in boxes if b.name not in self.delivered and slot_valid(slot_errors(b, xy, self.c), self.c)]
            match = valid[0] if len(valid) == 1 else None
            self.dwell[i] = self.dwell[i] + dt if match and match == self.candidate[i] else 0.
            self.candidate[i] = match
            self.occupied[i] = match if self.dwell[i] >= self.c['verification']['hold_s'] else None
        # Detect extra/tilted objects intersecting the three-slot loading zone.
        occupants = set(self.occupied) - {None}
        foreign = False
        for b in boxes:
            r = rotation(b.orientation_wxyz)
            extent = [sum(abs(r[i][j])*self.c['boxes']['size'][j]/2 for j in range(3)) for i in range(3)]
            if b.name not in occupants and any(
                all(abs(b.position[j]-xy[j]) < self.c['conveyor']['slot_size_xy'][j]/2+extent[j] for j in range(2))
                and abs(b.position[2]-self.c['conveyor']['top_z']) < extent[2]+.15
                for xy in self.c['conveyor']['slots_xy']):
                foreign = True
        if len(occupants) == 3 and safe_to_index and not foreign:
            self.batch = list(self.occupied)
            self.state, self.elapsed, self.distance = 'INDEXING', 0., 0.
            self.events.append({'event': 'index_started', 'boxes': list(self.batch)})
            return self.c['conveyor']['speed']
        return 0.


class TaskLoop:
    """Action contract for an external perception + manipulation implementation."""
    def __init__(self):
        self.phase = 'OBSERVE'
        self.last_action_success = None

    def decide(self, remaining, direct_candidates):
        if self.phase not in ('OBSERVE', 'DONE'):
            raise RuntimeError('Finish and verify the current action before re-planning')
        self.phase = 'DONE' if not remaining else ('DIRECT_PNP' if direct_candidates else 'UNLOCK')
        return self.phase

    def action_finished(self):
        if self.phase not in ('DIRECT_PNP', 'UNLOCK'): raise RuntimeError('No action in flight')
        self.phase = 'VERIFY'

    def verified(self, success):
        if self.phase != 'VERIFY': raise RuntimeError('No action awaiting verification')
        self.last_action_success = bool(success)
        self.phase = 'OBSERVE'
