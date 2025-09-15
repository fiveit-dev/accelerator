from typing import List, Optional, Union
from pydantic import BaseModel, Field


class Worklog(BaseModel):
    CREATEBY: Optional[str] = None
    CREATEDATE: Optional[str] = None
    DESCRIPTION: Optional[str] = None
    LOGTYPE: Optional[str] = None
    MODIFYBY: Optional[str] = None
    MODIFYDATE: Optional[str] = None
    SITEID: Optional[str] = None

    @classmethod
    def from_maximo_xml(cls, worklogs: Union[dict, List[dict]]) -> List:
        # Manejar el caso en que worklogs es un solo dict en lugar de una lista
        if isinstance(worklogs, dict):
            worklogs = [worklogs]

        return [
            cls(
                CREATEBY=wl.get("CREATEBY"),
                CREATEDATE=wl.get("CREATEDATE"),
                DESCRIPTION=wl.get("DESCRIPTION"),
                LOGTYPE=wl.get("LOGTYPE"),
                MODIFYBY=wl.get("MODIFYBY"),
                MODIFYDATE=wl.get("MODIFYDATE"),
                SITEID=wl.get("SITEID"),
            )
            for wl in worklogs
            if str(wl.get("CLIENTVIEWABLE")).lower in ("1", "True")
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
    WORKLOGS: List[Worklog] = Field(default_factory=list)

    @classmethod
    def from_maximo_xml(cls, ticket: dict):
        return cls(
            CLASS=ticket.get("CLASS"),
            DESCRIPTION=ticket.get("DESCRIPTION"),
            PLUSPCUSTOMER=ticket.get("PLUSPCUSTOMER"),
            STATUS=ticket.get("STATUS"),
            STATUSDATE=ticket.get("STATUSDATE"),
            TICKETID=ticket.get("TICKETID"),
            WORKLOGS=Worklog.from_maximo_xml(ticket.get("WORKLOG", [])),
        )

    @classmethod
    def from_maximo_dump(cls, ticket: dict):
        atributtes = ticket.get("Attributes", {})
        return cls(
            CLASS=atributtes.get("CLASS", {}).get("content"),
            DESCRIPTION=atributtes.get("DESCRIPTION", {}).get("content"),
            PLUSPCUSTOMER=atributtes.get("PLUSPCUSTOMER", {}).get("content"),
            STATUS=atributtes.get("STATUS", {}).get("content"),
            STATUSDATE=atributtes.get("STATUSDATE", {}).get("content"),
            TICKETID=atributtes.get("TICKETID", {}).get("content"),
            WORKLOGS=Worklog.from_maximo_dump(
                ticket.get("RelatedMbos", {}).get("WORKLOG", [])
            ),
        )
