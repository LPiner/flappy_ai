import attr
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
import time
from flappy_ai.types.keys import Keys
from flappy_ai.factories.selenium_key_factory import selenium_key_factory
from flappy_ai.models.image import Image
import numpy
import mss
from structlog import get_logger
import cv2

logger = get_logger(__name__)


@attr.s(auto_attribs=True)
class Game:
    _browser: webdriver = attr.ib(init=False)

    # X and Y positions of the game window.
    _pos_x: int = None
    _pos_y: int = None
    _browser_height: int = 830
    _browser_width: int = 570
    actions = 2 # Jump or Stay
    _game_over_button = cv2.imread("img/game_over.png", cv2.IMREAD_GRAYSCALE)

    def __attrs_post_init__(self):
        self._browser = webdriver.Firefox()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._browser:
            self._browser.close()

    def start_game(self):
        logger.debug("start_game")
        self._browser.set_window_size(self._browser_width, self._browser_height)
        self._browser.get('https://flappybird.io/')

        # Get our positions
        pos = self._browser.get_window_position()
        self._pos_x = pos["x"]
        self._pos_y = pos["y"]

        time.sleep(5)

    def _grab_screen(self) -> Image:
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

        #screen = pyscreenshot.grab(
        #    bbox=(self._pos_x,
        #          self._pos_y,
        #          self._pos_x + self._browser_height,
        #          self._pos_y + self._browser_width)
        #)
        # all the good shit is in cv2
        # Also turn it grayscale as we dont need the color data.
        #screen = cv2.cvtColor(numpy.array(screen), cv2.COLOR_BGR2GRAY)
        screen = numpy.array(screen)
        screen = screen[::4, ::4]
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        screen = Image(screen)
        #screen = self._preprocess_screen(screen)

        # Downscale the shit out of it.
        # Image processing is painfully slow and so in order to react fast enough we need to cut out anything that not important.
        # Here we are reducing the ratio by a factor of 3.
        #width = int(screen.shape[1] / 3)
        #height = int(screen.shape[0] / 3)
        #dim = (width, height)
        #screen = cv2.resize(screen, dim, interpolation=cv2.INTER_AREA)
        #screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

        #logger.debug("[_grab_screen] Exec time", runtime=time.time()-start_time)
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
            reward = -1
        else:
            reward = 1

        return screen.as_greyscale(), reward, done

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


