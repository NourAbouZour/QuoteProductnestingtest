# Nesting Verification and Scrap Factor Guide

## Overview
This guide explains how to verify that parts are being nested perfectly and that the scrap factor is correct when using the Nesting Center API.

## How Boards Are Handled

### Board Selection Priority
The system handles boards in the following order:

1. **User-Specified Boards (Priority 1)**: If you provide boards in the API request, these will be used
2. **Database Boards (Priority 2)**: If no boards in request, boards are loaded from database
3. **Default Fallback (Priority 3)**: If database fails, a default 3000×1500mm board is used

### Providing Boards in Request

You can explicitly provide boards when calling the nesting API:

```json
{
  "parts": [...],
  "boards": [
    {
      "id": "board_1",
      "width_mm": 3000,
      "height_mm": 1500,
      "quantity": 5
    },
    {
      "id": "board_2",
      "width_mm": 2500,
      "height_mm": 1250,
      "quantity": 10
    }
  ],
  "scrap_factor": 1.20
}
```

### What Gets Sent to API

**Both parts AND boards are sent to the Nesting Center API.** The API receives:
- Parts: List of parts with dimensions and quantities
- RawPlates: List of boards with dimensions and quantities
- Settings: Gap, margin, rotation, scrap factor

The logs will show exactly what's being sent:
```
📤 SENDING TO NESTING CENTER API:
PARTS (3 types):
  - part_1: 200×100mm × 5 pcs
  - part_2: 150×75mm × 10 pcs
  Total pieces: 15

BOARDS (2 types):
  - board_1: 3000×1500mm × 5 pcs
  - board_2: 2500×1250mm × 10 pcs
```

## How to Verify Nesting is Working Correctly

### Step 1: Check What Was Sent to API

Before nesting starts, you'll see a detailed log showing exactly what's being sent:

```
📤 READY TO SEND TO NESTING CENTER API
================================================================================

📋 PARTS TO SEND (3 types, 35 total pieces):
  1. part_1: 200.00×100.00mm × 5pcs = 100,000 mm²
  2. part_2: 150.00×75.00mm × 10pcs = 112,500 mm²
  3. part_3: 100.00×50.00mm × 20pcs = 100,000 mm²

📐 BOARDS TO SEND (2 types):
  1. board_1: 3000×1500mm × 5pcs = 4,500,000 mm² each
  2. board_2: 2500×1250mm × 10pcs = 3,125,000 mm² each

⚙️  SETTINGS:
  • Gap between parts: 5.0 mm
  • Margin from edges: 5.0 mm
  • Scrap factor: 1.2000 (20.00% waste)
  • Rotation: Fixed90
  • Timeout: 60 seconds
```

**Verify:**
- ✅ All parts are listed with correct dimensions and quantities
- ✅ All boards are listed with correct dimensions and quantities
- ✅ Settings match what you expect (gap, margin, scrap factor)

### Step 2: Verify Request Structure

After conversion, the system logs the API request structure:

```
🔍 VERIFYING API REQUEST STRUCTURE:
  ✓ Settings included: ['DistancePartPart', 'DistancePartRawPlate', 'RotationControl', 'ScrapFactor']
  ✓ ScrapFactor in settings: True
    ScrapFactor value: 1.2
  ✓ Parts in request: 3 types
  ✓ RawPlates in request: 2 types
```

**Verify:**
- ✅ ScrapFactor is in the settings
- ✅ Parts count matches your input
- ✅ RawPlates (boards) count matches your input

### Step 3: Check Nesting Result

After nesting completes, you'll see how parts were actually nested:

```
📊 NESTING RESULT SUMMARY:
Success: True
Total plates used: 1
Total parts nested: 35
Layouts returned: 1

📐 LAYOUT DETAILS:
  Plate 1 (index 0):
    - Parts nested: 35
    - Parts area: 312,500 mm²
    - Board area used: 4,500,000 mm²
    - Utilization: 6.94%
    - Scrap: 93.06%
    - Parts on plate:
      1. part_1: 200×100mm
      2. part_2: 150×75mm
      ... and 33 more parts
```

**Verify:**
- ✅ All parts were nested (compare "Total parts nested" with expected quantity)
- ✅ Utilization and scrap percentages make sense
- ✅ Parts listed match what you sent

### Step 4: Verify Scrap Factor Accuracy

Compare the expected vs actual scrap:

```
📈 OVERALL STATISTICS:
  • Overall utilization: 82.15%
  • Overall scrap: 17.85%
  • Expected scrap (from factor 1.20): 20.00%
  • Scrap difference: 2.15%
```

**Verify:**
- ✅ Scrap difference is small (< 5% is good, < 10% is acceptable)
- ✅ If difference is large (> 15%), check:
  - Parts may not fit optimally on boards
  - Margins/gaps may be affecting calculations
  - Some parts may not have been nested

## Part Validation

### Before Sending to API
The system now validates all parts before sending them to the Nesting Center API:

1. **Dimension Validation**
   - Checks that length and width are > 0
   - Warns if dimensions are < 1mm (too small)
   - Rounds dimensions to 2 decimal places for precision

2. **Quantity Validation**
   - Ensures quantity is a positive integer
   - Warns if quantity is > 10,000 (verify this is correct)
   - Defaults to 1 if quantity is missing or invalid

3. **Logging**
   All parts are logged before sending:
   ```
   ✅ Part 1: 200.00x100.00mm × 5 pieces (Area: 20000.00 mm²)
   ✅ Part 2: 150.00x75.00mm × 10 pieces (Area: 11250.00 mm²)
   ```

### What to Check
- **Total Part Types**: Should match the number of unique parts in your DXF
- **Total Part Pieces**: Should match the sum of all quantities
- **Part Areas**: Verify dimensions look correct
- **Validation Warnings**: Review any warnings about invalid dimensions or quantities

## Scrap Factor Verification

### Understanding Scrap Factor
- **Scrap Factor 1.0** = No waste (100% utilization)
- **Scrap Factor 1.20** = 20% waste (80% utilization)
- **Scrap Factor 1.50** = 50% waste (66.7% utilization)

Formula: `Waste Percentage = (Scrap Factor - 1.0) × 100`

### How Scrap Factor is Used

1. **Sent to API**
   - The scrap factor is included in the Nesting Center API request settings
   - Look for: `"ScrapFactor": 1.20` in the request

2. **Validation**
   - System warns if scrap factor < 1.0 (negative waste)
   - System warns if scrap factor > 2.0 (>100% waste)

3. **Result Verification**
   After nesting completes, the system compares:
   - **Expected Scrap**: Calculated from the scrap factor you sent
   - **Actual Scrap**: Calculated from the nesting result (used area vs parts area)

### How to Verify Scrap Factor is Correct

#### Step 1: Check the Logs
After running nesting, look for this section:
```
📈 OVERALL STATISTICS:
  • Total board area used: 9,000,000.00 mm² (9.0000 m²)
  • Total parts area nested: 7,200,000.00 mm² (7.2000 m²)
  • Overall utilization: 80.00%
  • Overall scrap: 20.00%
  • Expected scrap (from factor 1.20): 20.00%
  • Scrap difference: 0.00%
```

#### Step 2: Verify Calculations
- **Utilization** = (Parts Area / Board Area Used) × 100
- **Scrap Percentage** = 100% - Utilization
- **Expected Scrap** = (Scrap Factor - 1.0) × 100

The scrap difference should be small (< 5%). If it's larger, investigate:
- Parts may not fit perfectly on boards
- API may have adjusted for margins/gaps
- Some parts may not have been nested

#### Step 3: Check Per-Plate Results
For each plate, verify:
```
✅ Plate 1: 3000x1500mm, 25 parts, Utilization: 82.45%, Scrap: 17.55%
```

If scrap differs significantly from expected:
- Check if all parts fit on the plate
- Verify margins and gaps are correct
- Review if parts are rotated optimally

## Nesting Completeness Verification

### Verify All Parts Were Nested

After nesting, check:
```
✅ All parts nested successfully!
   All 50 pieces were nested correctly
```

Or if incomplete:
```
⚠️  WARNING: Not all parts were nested!
   Expected: 50 pieces
   Nested: 48 pieces
   Missing: 2 pieces
```

### What to Do If Parts Are Missing

1. **Check Board Sizes**
   - Verify boards are large enough for all parts
   - Consider using larger boards or multiple board sizes

2. **Check Part Dimensions**
   - Ensure parts aren't larger than boards (accounting for margins)
   - Verify dimensions are correct

3. **Check Quantities**
   - Ensure quantities match what you expect
   - Review if some parts should be excluded

4. **Review Logs**
   - Check for validation warnings
   - Look for parts that were skipped

## API Request Verification

### What Gets Sent to API

The system sends:
```json
{
  "Context": {
    "Settings": {
      "DistancePartPart": 5.0,
      "DistancePartRawPlate": 5.0,
      "RotationControl": "Fixed90",
      "ScrapFactor": 1.20
    },
    "Problem": {
      "Parts": [
        {
          "RectangularShape": {
            "Length": 200.0,
            "Width": 100.0
          },
          "Quantity": 5,
          "Label": "part_1"
        }
      ],
      "RawPlates": [...]
    }
  },
  "StopJson": {
    "AllPartsNested": true,
    "Timeout": 60
  }
}
```

### Verify Request Settings

Check logs for:
```
⚙️  NESTING SETTINGS:
  • Gap between parts: 5.0 mm
  • Margin from edges: 5.0 mm
  • Scrap factor: 1.2000 (20.00% waste)
  • Rotation: Fixed90 (0°, 90°, 180°, 270°)
  • Timeout: 60 seconds
```

## Troubleshooting

### Problem: Scrap factor doesn't match result

**Possible Causes:**
1. API may calculate scrap differently (includes margins/gaps)
2. Parts may not fit perfectly, causing different utilization
3. The scrap factor in the API may be informational only

**Solution:**
- Review the actual scrap percentage from results
- Compare actual vs expected (small differences are normal)
- If difference is large (>10%), verify parts dimensions and board sizes

### Problem: Not all parts nested

**Possible Causes:**
1. Parts too large for available boards
2. Quantities incorrect
3. Board sizes insufficient

**Solution:**
1. Check board dimensions vs part dimensions
2. Verify quantities are correct
3. Try using larger boards or adjust margins/gaps
4. Review validation warnings in logs

### Problem: Parts dimensions seem wrong

**Solution:**
1. Check DXF file for correct dimensions
2. Verify bounding box calculations
3. Review part_images data for correct dimensions
4. Check if parts are being rotated correctly

## Best Practices

1. **Always Review Logs**
   - Check part validation summary
   - Review nesting statistics
   - Verify scrap factor usage

2. **Verify Inputs First**
   - Check part dimensions before nesting
   - Verify quantities are correct
   - Ensure board sizes are appropriate

3. **Monitor Scrap Differences**
   - Small differences (<5%) are normal
   - Large differences (>10%) need investigation
   - Compare expected vs actual regularly

4. **Check Completeness**
   - Always verify all parts were nested
   - Review any missing parts
   - Ensure quantities match expectations

## Example: Good Nesting Result

```
📋 PART VALIDATION - Verifying all parts before sending to Nesting Center API
================================================================================
  ✅ Part 1: 200.00x100.00mm × 5 pieces (Area: 100000.00 mm²)
  ✅ Part 2: 150.00x75.00mm × 10 pieces (Area: 112500.00 mm²)
  ✅ Part 3: 100.00x50.00mm × 20 pieces (Area: 100000.00 mm²)

📊 VALIDATION SUMMARY:
  • Total part types: 3
  • Total part pieces: 35
  • Total parts area: 312,500.00 mm² (0.3125 m²)
  ✅ All parts validated successfully

📐 BOARD VALIDATION:
  ✅ Board 1: 3000x1500mm (Area: 4,500,000.00 mm²), Qty: 10

⚙️  NESTING SETTINGS:
  • Gap between parts: 5.0 mm
  • Margin from edges: 5.0 mm
  • Scrap factor: 1.2000 (20.00% waste)
  • Rotation: Fixed90

🚀 CALLING NESTING CENTER CLOUD API...

📥 NESTING RESULT RECEIVED FROM API
================================================================================
  Success: True
  Total plates used: 1
  Total parts nested: 35

✅ All parts nested successfully!
   All 35 pieces were nested correctly

📊 LAYOUT ANALYSIS:
  ✅ Plate 1: 3000x1500mm, 35 parts, Utilization: 82.15%, Scrap: 17.85%

📈 OVERALL STATISTICS:
  • Total board area used: 4,500,000.00 mm² (4.5000 m²)
  • Total parts area nested: 3,696,750.00 mm² (3.6968 m²)
  • Overall utilization: 82.15%
  • Overall scrap: 17.85%
  • Expected scrap (from factor 1.20): 20.00%
  • Scrap difference: 2.15%
```

This shows a successful nesting with all parts nested and scrap factor within expected range.
