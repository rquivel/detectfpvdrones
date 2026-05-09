#!/bin/bash
shopt -s nullglob
for i in *.webp; do sips -s format png "$i" --out "${i%.webp}.png"; done
for i in *.jpeg; do sips -s format png "$i" --out "${i%.jpeg}.png"; done
for i in *.avif; do sips -s format png "$i" --out "${i%.avif}.png"; done
rm -f *.webp *.jpeg *.avif