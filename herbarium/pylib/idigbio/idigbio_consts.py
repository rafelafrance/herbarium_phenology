"""Literals related to iDigBio files."""
COLUMN = "accessuri"

OCCURRENCE = """ coreid dwc:basisOfRecord dwc:kingdom dwc:phylum dwc:class
    dwc:order dwc:family dwc:genus dwc:specificEpithet dwc:scientificName
    dwc:eventDate dwc:continent dwc:country dwc:stateProvince dwc:county dwc:locality
    idigbio:geoPoint
    """.split()

OCCURRENCE_RAW = """ coreid dwc:reproductiveCondition dwc:occurrenceRemarks
    dwc:dynamicProperties dwc:fieldNotes
    """.split()

MULTIMEDIA = """ coreid accessURI """.split()
