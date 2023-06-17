
class Nightlight:

    def read_night_light_data(self, product):
        """
        Reads raw nightlight data from LAADS using a list of tiles extracted beforehand.
        :return:
        """

    def preprocess_nightlight(self):
        """
        Projects and clips the tiles to desired projection system and study area
        :return:
        """

    def satellite_angle_corrections(self):
        """
        If needed applies satellite Azimuth and Zenith corrections.
        :return:
        """

    def nightlight_products(self, corrected, interval):
        """
        Creates daily, weekly, monthly, annual tiles with parametric statistics (average and median)
        :return:
        """


