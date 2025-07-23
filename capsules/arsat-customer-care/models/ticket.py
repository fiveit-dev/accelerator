from typing import List, Optional
from pydantic import BaseModel


class Worklog(BaseModel):
    CREATEBY: Optional[str] = None
    CREATEDATE: Optional[str] = None
    DESCRIPTION: Optional[str] = None
    LOGTYPE: Optional[str] = None
    LOGDESCRIPTION: Optional[str] = None
    MODIFYBY: Optional[str] = None
    MODIFYDATE: Optional[str] = None
    SITEID: Optional[str] = None

    @classmethod
    def from_maximo_oslc(cls, worklogs: List[dict]) -> List:
        return [
            cls(
                CREATEBY=wl.get("spi:createby"),
                CREATEDATE=wl.get("spi:createdate"),
                DESCRIPTION=wl.get("spi:description"),
                LOGTYPE=wl.get("spi:logtype"),
                LOGDESCRIPTION=wl.get("spi:logdescription"),
                MODIFYBY=wl.get("spi:modifyby"),
                MODIFYDATE=wl.get("spi:modifydate"),
                SITEID=wl.get("spi:siteid"),
            )
            for wl in worklogs
            if wl.get("spi:clientviewable", False)
        ]

    @classmethod
    def from_maximo_dump(cls, worklogs: List[dict]) -> List:
        return [
            cls(
                CREATEBY=wl.get("Attributes", {}).get("CREATEBY", {}).get("content"),
                CREATEDATE=wl.get("Attributes", {})
                .get("CREATEDATE", {})
                .get("content"),
                DESCRIPTION=wl.get("Attributes", {})
                .get("DESCRIPTION", {})
                .get("content"),
                LOGTYPE=wl.get("Attributes", {}).get("LOGTYPE", {}).get("content"),
                LOGDESCRIPTION=wl.get("Attributes", {})
                .get("LOGDESCRIPTION", {})
                .get("content"),
                MODIFYBY=wl.get("Attributes", {}).get("MODIFYBY", {}).get("content"),
                MODIFYDATE=wl.get("Attributes", {})
                .get("MODIFYDATE", {})
                .get("content"),
                SITEID=wl.get("Attributes", {}).get("SITEID", {}).get("content"),
            )
            for wl in worklogs
            if wl.get("Attributes", {}).get("CLIENTVIEWABLE", {}).get("content", False)
        ]


class Ticket(BaseModel):
    CLASS: Optional[str] = None
    DESCRIPTION: Optional[str] = None
    PLUSPCUSTOMER: Optional[str] = None
    STATUS: Optional[str] = None
    STATUSDATE: Optional[str] = None
    TICKETID: Optional[str] = None
    WORKLOGS: List[Worklog] = []

    @classmethod
    def from_maximo_oslc(cls, ticket: dict):
        return cls(
            CLASS=ticket.get("spi:class"),
            DESCRIPTION=ticket.get("spi:description"),
            PLUSPCUSTOMER=ticket.get("spi:pluspcustomer"),
            STATUS=ticket.get("spi:status"),
            STATUSDATE=ticket.get("spi:statusdate"),
            TICKETID=ticket.get("spi:ticketid"),
            WORKLOGS=Worklog.from_maximo_oslc(ticket.get("spi:worklog", [])),
        )

    @classmethod
    def from_maximo_dump(cls, ticket: dict):
        return cls(
            CLASS=ticket.get("Attributes", {}).get("CLASS", {}).get("content"),
            DESCRIPTION=ticket.get("Attributes", {})
            .get("DESCRIPTION", {})
            .get("content"),
            PLUSPCUSTOMER=ticket.get("Attributes", {})
            .get("PLUSPCUSTOMER", {})
            .get("content"),
            STATUS=ticket.get("Attributes", {}).get("STATUS", {}).get("content"),
            STATUSDATE=ticket.get("Attributes", {})
            .get("STATUSDATE", {})
            .get("content"),
            TICKETID=ticket.get("Attributes", {}).get("TICKETID", {}).get("content"),
            WORKLOGS=Worklog.from_maximo_dump(
                ticket.get("RelatedMbos", {}).get("WORKLOG", [])
            ),
        )
