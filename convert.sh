#!/bin/bash
for i in *.webp; do sips -s format png "$i" --out "${i%.webp}.png"; done
for i in *.jpeg; do sips -s format png "$i" --out "${i%.jpeg}.png"; done
rm *.webp *.jpeg