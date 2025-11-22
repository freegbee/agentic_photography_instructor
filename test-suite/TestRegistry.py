
from juror_client.registry import register_service, get_registered_service, clear_registry, get_juror_service
from juror_client.juror_cache import JurorCachingService
from juror_client.juror_service import JurorService
from juror_shared.models_v1 import ScoringResponsePayloadV1


class DummyService(JurorService):
    def score_image(self, image_path: str):
        return ScoringResponsePayloadV1(filename=image_path, score=0.1)

    def score_ndarray(self, array, filename=None, encoding: str = "npy"):
        return ScoringResponsePayloadV1(filename=filename, score=0.2)

    def close(self) -> None:
        pass


def test_register_and_get_and_clear():
    clear_registry()

    assert get_registered_service("nope") is None

    s = DummyService()
    register_service("dummy", s)
    got = get_registered_service("dummy")
    assert got is s

    clear_registry()
    assert get_registered_service("dummy") is None


def test_get_juror_service_registers_when_name_and_not_given_service():
    # Ensure registry is clean
    clear_registry()
    # Create with name -> registry should contain it
    svc = get_juror_service(name="reg_test_service", use_cache=True)
    reg = get_registered_service("reg_test_service")
    assert reg is not None
    # If we asked for a cached service, it should be wrapped by JurorCachingService
    assert isinstance(svc, JurorCachingService)

    clear_registry()

