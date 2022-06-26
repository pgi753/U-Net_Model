from arena_server.wifi_network_operator import WiFiNetworkOperator
from arena_server.mobility import WiFiRandomWaypointMobility
from arena_server.wifi_parameter import WiFiParam
from arena_server.simulator import Simulator
from arena_server.visualization import Visualization
import numpy as np
from threading import Thread
from arena_server.wifi_bot_controller import WiFiCSMAController, WiFiPeriodicController
from arena_server.remote_player_manager import RemotePlayerManager
# from arena_server.trained_model_controller import TrainedController

num_frequency_channel = 4
freq_channel_list = np.arange(num_frequency_channel).tolist()
center_freq_list = WiFiParam.FREQUENCY_LIST_10MHZ[0: num_frequency_channel]
sim = Simulator(center_freq_list=center_freq_list)
sim.logger.add_print_handler()
sim.logger.add_file_handler('output.log')
sim.logger.set_clock(clock_interval=10)
vis = Visualization(refresh_rate=30, max_num_line_object=100, plane_size=(40, 40), node_size=0.4,
                    activate_time_domain_view=True)
sim.logger.add_handler(vis)


# Network Operator 1 (Remote)
w1 = WiFiNetworkOperator(name='w1', num_sta=20, freq_channel_list=freq_channel_list)
m1 = WiFiRandomWaypointMobility(area_center=(5, 5), area_radius=15, speed_range=(30, 50),
                                avg_pause_time=3, update_interval=-1)
w1.set_mobility(m1)
sim.add_network_operator(w1)

# Network Operator 2(local)
w2 = WiFiNetworkOperator(name='w2', num_sta=20, freq_channel_list=freq_channel_list)
m2 = WiFiRandomWaypointMobility(area_center=(-5, -5), area_radius=15, speed_range=(30, 50),
                                avg_pause_time=3, update_interval=-1)
w2.set_mobility(m2)
c2 = WiFiPeriodicController(num_back_off=100, num_unit_packet=1, frequency_channel_list=[0, 2])
w2.set_controller(c2)
sim.add_network_operator(w2)

# Network Operator 3(local)
w3 = WiFiNetworkOperator(name='w3', num_sta=20, freq_channel_list=freq_channel_list)
m3 = WiFiRandomWaypointMobility(area_center=(0, 0), area_radius=15, speed_range=(30, 50),
                                avg_pause_time=3, update_interval=-1)
w3.set_mobility(m3)
c3 = WiFiPeriodicController(num_back_off=50, num_unit_packet=2, frequency_channel_list=[1, 3])
w3.set_controller(c3)
sim.add_network_operator(w3)


# Make remote connection
remote_player_manager = RemotePlayerManager()
remote_player_manager.set_server_socket('127.0.0.1', 8000)
remote_player_manager.connect_to_players({'p1': w1})

# Start simulation
thread = Thread(target=sim.run, args=(1000000000,))
thread.start()
