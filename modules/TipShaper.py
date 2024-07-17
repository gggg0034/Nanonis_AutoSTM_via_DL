# TipShaper.py by CoccaGuo at 2022/05/21 18:19

import logging
import time
from core import NanonisController, Operate


class TipShaper(Operate):
    '''
    TipShaper in nanonis api is not available, so we use the following code to simulate it.
    '''

    def __init__(self, session: NanonisController, bias=3, tip_lift='-300p'):
        super().__init__(session)
        self.bias = self.session.try_convert(bias)
        self.tip_lift = self.session.try_convert(tip_lift)

    def safety_check(self):
        if self.session.isZCtrlWork():
            return True
        else:
            return False

    def _operate(self):
        s = self.session
        z_c = s.TipZGet()
        b_c = s.BiasGet()
        s.ZCtrlOnOffSet(False)
        logging.info('TipShaper: bias {}, tip lift {}.'.format(
            self.bias, self.tip_lift))
        time.sleep(1)
        s.TipZSet(z_c+self.tip_lift)
        s.BiasSet(self.bias)
        time.sleep(0.1)
        s.TipZSet(z_c)
        s.BiasSet(b_c)
        s.ZCtrlOnOffSet(True)
        time.sleep(0.5)
