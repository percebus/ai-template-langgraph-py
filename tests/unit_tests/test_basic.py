from hamcrest import assert_that, is_


def test_True_is_True() -> None:
    assert_that(True, is_(True))
