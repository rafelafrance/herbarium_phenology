"""Tag records as flowering, fruiting, or leaf-out.

We want both positive and negative cases. This is done after we have identified
the records as angiosperms.
"""

FLOWER = """ flowers? petals? corollas? tepals? rays? sepals? inflorescences?
    flores fls? infl inflor stigmas? stigmata """.split()
NO_FLOWER = """  """.split()

FRUIT = """ fruits? seed? fruta """.split()
NO_FRUIT = """ sterile """.split()

LEAF = """ leaf leaves bracts? foliage folhas hoja """.split()
NO_LEAF = """  """.split()

ALL_TAGS = FLOWER + FRUIT + LEAF + NO_FLOWER + NO_FRUIT + NO_LEAF
