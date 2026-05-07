"""
test_env_sanity.py - Run this in your project root to verify the env works before retraining.
Usage: rl-env\Scripts\python.exe test_env_sanity.py
"""
import sys
sys.path.insert(0, ".")

from src.environment.warehouse_env import WarehouseEnv
from src.environment.grid_world import CellType
from src.environment.robot import Action

env = WarehouseEnv({"seed": 42, "max_steps": 200})
obs, infos = env.reset()

print("=== ENV SANITY CHECK ===\n")

# Check shelf walkability
shelf_positions = env.grid.shelf_positions
print(f"Total shelves: {len(shelf_positions)}")
print(f"Shelf at {shelf_positions[0]} is walkable: {env.grid.is_walkable(shelf_positions[0])}")

# Check robot states
print(f"\nInitial robot states:")
for agent_id, robot in env.robots.items():
    order_info = f"order→shelf@{robot.assigned_order.shelf_position}" if robot.assigned_order else "no order"
    print(f"  {agent_id}: pos={robot.position}, state={robot.state.name}, {order_info}")

# Manually walk robot_0 to its shelf and attempt PICK
robot = env.robots["robot_0"]
if robot.assigned_order:
    shelf = robot.assigned_order.shelf_position
    print(f"\nManually teleporting robot_0 to shelf {shelf} to test PICK...")
    
    # Teleport
    env.grid.clear_robot_position(robot.position)
    robot.position = shelf
    env.grid.set_robot_position(robot.robot_id, shelf)
    
    print(f"  robot_0 now at {robot.position}, shelf_position={shelf}, match={robot.position == shelf}")
    
    # Force PICK action
    from src.environment.robot import RobotState
    events = {"deliveries": [], "collisions": [], "missed_deadlines": [], 
              "new_positions": {}, "goal_positions": {}, "energy_waste": []}
    env._process_pick(robot, events)
    
    print(f"  After PICK: carrying={robot.carrying_item}, state={robot.state.name}")
    
    if robot.carrying_item:
        print("\n  ✓ PICK WORKS! Environment is functional.")
        
        # Now teleport to delivery zone and test DELIVER
        delivery = robot.assigned_order.delivery_position
        env.grid.clear_robot_position(robot.position)
        robot.position = delivery
        env.grid.set_robot_position(robot.robot_id, delivery)
        
        env._process_deliver(robot, events)
        print(f"  After DELIVER: carrying={robot.carrying_item}, deliveries={robot.total_deliveries}")
        print(f"  Events deliveries: {events['deliveries']}")
        
        if robot.total_deliveries > 0:
            print("\n  ✓ DELIVER WORKS! Ready to retrain.")
        else:
            print("\n  ✗ DELIVER FAILED. Check _process_deliver logic.")
    else:
        print("\n  ✗ PICK FAILED even at correct shelf. Deeper bug exists.")
        print(f"  robot.assigned_order={robot.assigned_order}")
        print(f"  robot.position={robot.position}")
        print(f"  shelf_position={robot.assigned_order.shelf_position if robot.assigned_order else 'N/A'}")
else:
    print("\nrobot_0 has no order assigned - check _assign_available_orders")

print("\n=== END SANITY CHECK ===")