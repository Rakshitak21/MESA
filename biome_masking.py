import torch, re

BIOMES = ["forest", "desert", "mountain", "grassland", "water"]

def parse_biome_prompt(prompt: str):
    prompt = prompt.lower()
    parsed = {}

    # percentages → "70% desert 30% forest"
    matches = re.findall(r"(\d+)%\s*(\w+)", prompt)
    if matches:
        for pct, biome in matches:
            if biome in BIOMES:
                parsed[biome] = float(pct) / 100.0
        return parsed

    # left/right
    if "left" in prompt and "right" in prompt:
        left = next((b for b in BIOMES if b in prompt.split("left")[1]), None)
        right = next((b for b in BIOMES if b in prompt.split("right")[1]), None)
        return {"left": left, "right": right}

    # center/around
    if "center" in prompt and "around" in prompt:
        center = next((b for b in BIOMES if b in prompt.split("center")[1]), None)
        around = next((b for b in BIOMES if b in prompt.split("around")[1]), None)
        return {"center": center, "around": around}

    # single biome
    for biome in BIOMES:
        if biome in prompt:
            return {biome: 1.0}

    return {}


def make_biome_mask(H, W, prompt):
    parsed = parse_biome_prompt(prompt)
    mask = torch.zeros(1, len(BIOMES), H, W)

    if all(isinstance(v, float) for v in parsed.values()):
        for biome, weight in parsed.items():
            idx = BIOMES.index(biome)
            mask[:, idx] = weight
        return mask

    if "left" in parsed and "right" in parsed:
        if parsed["left"]:
            idx = BIOMES.index(parsed["left"])
            mask[:, idx, :, :W//2] = 1
        if parsed["right"]:
            idx = BIOMES.index(parsed["right"])
            mask[:, idx, :, W//2:] = 1
        return mask

    if "center" in parsed and "around" in parsed:
        cx1, cx2 = W//4, 3*W//4
        cy1, cy2 = H//4, 3*H//4
        if parsed["center"]:
            mask[:, BIOMES.index(parsed["center"]), cy1:cy2, cx1:cx2] = 1
        if parsed["around"]:
            idx = BIOMES.index(parsed["around"])
            mask[:, idx] = 1
            mask[:, idx, cy1:cy2, cx1:cx2] = 0
        return mask

    if len(parsed) == 1:
        biome = next(iter(parsed))
        mask[:, BIOMES.index(biome)] = 1
        return mask

    return mask
