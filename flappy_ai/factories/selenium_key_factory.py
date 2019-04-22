from flappy_ai.types.keys import Keys
from selenium.webdriver.common.keys import Keys as SeleniumKeys


def selenium_key_factory(key: Keys) -> SeleniumKeys:
    if key is Keys.UP:
        return SeleniumKeys.UP
    elif key is Keys.DOWN:
        return SeleniumKeys.DOWN
    elif key is Keys.LEFT:
        return SeleniumKeys.LEFT
    elif key is Keys.RIGHT:
        return SeleniumKeys.RIGHT
    elif key is Keys.ENTER:
        return SeleniumKeys.RETURN
    elif key is Keys.SPACE:
        return SeleniumKeys.SPACE
    else:
        raise NotImplementedError
