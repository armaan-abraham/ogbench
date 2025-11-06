from agents.crl import CRLAgent
from agents.crl_actionless import CRLAgentActionless
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.gcivl_actionless import GCIVLAgentActionless
from agents.vcrl import VCRLAgent
from agents.hiql import HIQLAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent

agents = dict(
    crl_actionless=CRLAgentActionless,
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl_actionless=GCIVLAgentActionless,
    hiql=HIQLAgent,
    qrl=QRLAgent,
    sac=SACAgent,

    vcrl=VCRLAgent,
)
