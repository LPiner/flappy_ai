import attr
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.webdriver import FirefoxWebElement
import time
from flappy_ai.types.keys import Keys
from flappy_ai.factories.selenium_key_factory import selenium_key_factory
from flappy_ai.models.image import Image
import numpy
import mss
from structlog import get_logger
import numpy as np
import cv2

logger = get_logger(__name__)


@attr.s(auto_attribs=True)
class Game:
    headless: bool = attr.ib(default=False)
    _game_over: bool =attr.ib(init=False, default=False)
    _browser: webdriver = attr.ib(init=False)
    _game_element: FirefoxWebElement = attr.ib(init=False, default=None)

    # X and Y positions of the game window.
    _pos_x: int = None
    _pos_y: int = None
    _browser_height: int = 830
    _browser_width: int = 570
    _game_over_button = cv2.imread("img/game_over.png", cv2.IMREAD_GRAYSCALE)

    @staticmethod
    def actions():
        return 2

    def quit(self):
        if self._browser:
            self._browser.close()

    def __enter__(self):
        options = Options()
        options.headless = self.headless
        self._browser = webdriver.Firefox(options=options)
        self._browser.set_window_size(self._browser_width, self._browser_height)
        self._browser.get('https://flappybird.io/')
        self._game_element = self._browser.find_element_by_id(id_="testCanvas")
        pos = self._browser.get_window_position()
        self._pos_x = pos["x"]
        self._pos_y = pos["y"]
        time.sleep(3)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._browser:
            self._browser.close()

    def _grab_screen(self) -> Image:
        image = np.fromstring(self._game_element.screenshot_as_png, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = image[::4, ::4]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #import matplotlib.pyplot as plt
        #plt.imshow(image)
        #plt.show()
        return Image(image)

    def _grab_screen_legacy(self) -> Image:
        """
        Grab a screenshot of the game.
        """
        start_time = time.time()

        # In flappy bird half the screen is worthless so we can just filter it out.
        start_x = self._pos_x + 165
        start_y = self._pos_y + 35
        region = {
            "top": start_x,
            "left": start_y + 160,
            "width": start_x + 155,
            "height": start_y + 595,
        }
        with mss.mss() as sct:
            screen = sct.grab(region)

        screen = numpy.array(screen)
        screen = screen[::4, ::4]
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        screen = Image(screen)
        return screen

    def step(self, action) -> (numpy.array, int, bool):
        if action == 0:
            pass
        elif action == 1:
            self.input(Keys.SPACE)

        screen = self._state()

        def game_over(screen: Image):

            # Find colors that only the bird is going to have.
            mask = cv2.inRange(screen.as_HSV(), lowerb=numpy.array([0, 100, 100]), upperb=numpy.array([15, 255, 255]))
            points = cv2.findNonZero(mask)
            # merge our findings together.

            # Bird is off screen?
            if points is None or points.size == 0:
                return True

            # Template matching is very good for exact matches.
            match = cv2.matchTemplate(screen.as_greyscale(), self._game_over_button, cv2.TM_CCOEFF_NORMED)
            match = numpy.where(match >= 0.8)
            #logger.debug("[find_match_with_template]", runtime=time.time() - start_time, m=match)
            if match is None or not match[0].size:
                return False
            return True

        done = int(game_over(screen))
        if done:
            self._game_over = True

        if done:
            reward = -1
        else:
            reward = 1

        return screen.as_greyscale(), reward, done

    def game_over(self) -> bool:
        return self._game_over

    def reset(self):
        self.input(Keys.SPACE)
        self.input(Keys.SPACE)
        time.sleep(.5)

    def _state(self) -> Image:
        return self._grab_screen()

    def input(self, key: Keys):
        key = selenium_key_factory(key=key)
        #element = self._browser.find_element_by_tag_name("canvas")
        actions = ActionChains(driver=self._browser)
        actions.send_keys(key).perform()

