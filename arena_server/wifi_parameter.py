# https://kr.mathworks.com/help/wlan/ug/802-11ax-phy-focused-system-level-simulation.html;jsessionid=30c78f508f7771d4e6b5289a92eb
# https://kr.mathworks.com/help/wlan/ug/802-11ax-packet-error-rate-simulation-for-single-user-format.html
# https://www.semfionetworks.com/blog/mcs-table-updated-with-80211ax-data-rates
# http://ieeexplore.ieee.org.ssl.sa.skku.edu:8080/document/8465732


class WiFiParam:
    SIFS = 16  # us
    SLOT_TIME = 9  # us
    SYMBOL_DURATION = 13.6  # us, 0.8 us GI (guard interval)
    UNIT_CHANNEL_BANDWIDTH = 10  # MHz
    NUM_SUBCARRIER_10MHZ = 106
    BITS_PER_SUBCARRIER = 2  # QPSK
    CODE_RATE = 1/2
    SYMBOL_PER_UNIT_PACKET = 16
    DATA_BYTES_PER_UNIT_PACKET = NUM_SUBCARRIER_10MHZ * BITS_PER_SUBCARRIER * CODE_RATE * SYMBOL_PER_UNIT_PACKET / 8
    UNIT_PACKET_DURATION = SYMBOL_PER_UNIT_PACKET * SYMBOL_DURATION
    AP_TX_POWER = 100  # mW
    STA_TX_POWER = 32  # mW
    AP_ANTENNA_GAIN = 0  # dBi
    STA_ANTENNA_GAIN = -2  # dBi
    NOISE_FIGURE = 7  # dB
    TEMPERATURE = 290  # Kelvin
    BOLTZMANN = 1.380649 * (10 ** (-23))  # J/Kelvin
    NOISE_POWER = BOLTZMANN * TEMPERATURE * (UNIT_CHANNEL_BANDWIDTH * (10 ** 6)) * (10 ** 3) \
                  * (10 ** (NOISE_FIGURE / 10))  # mW
    SINR_THRESHOLD = 20  # dB
    SYMBOL_PER_ACK = 2
    ACK_DURATION = SYMBOL_PER_ACK * SYMBOL_DURATION
    CCA_THRESHOLD = -70  # dBm
    FREQUENCY_LIST_10MHZ = [5930 + 5 * ind for ind in range(96)]        # 5925~6425MHz
