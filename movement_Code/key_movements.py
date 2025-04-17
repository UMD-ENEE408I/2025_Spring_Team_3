#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
import sys, select, termios, tty

# Movement bindings
moveBindings = {
    'w': (1, 0),
    's': (-1, 0),
    'a': (0, 1),
    'd': (0, -1),
}

# Speed bindings
speedBindings = {
    'q': (1.1, 1.1),
    'z': (0.9, 0.9),
}

def getKey():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

speed = 0.5
turn = 1.0
x = 0
th = 0

settings = termios.tcgetattr(sys.stdin)

rospy.init_node('turtlebot3_teleop_keyboard')
pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

try:
    print("Reading from keyboard and Publishing to cmd_vel")
    while not rospy.is_shutdown():
        key = getKey()
        if key in moveBindings:
            x, th = moveBindings[key]
        elif key in speedBindings:
            speed *= speedBindings[key][0]
            turn *= speedBindings[key][1]
        elif key == '\x03':  # Ctrl-C
            break

        twist = Twist()
        twist.linear.x = x * speed
        twist.angular.z = th * turn
        pub.publish(twist)

except Exception as e:
    print(e)

finally:
    twist = Twist()
    pub.publish(twist)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
