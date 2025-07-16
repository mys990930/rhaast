from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    @abstractmethod
    def generate_signal(self, candles):
        pass

    @abstractmethod
    def get_stop_loss(self, candles):
        """ R 값을 기반으로 손절가 반환 """
        pass