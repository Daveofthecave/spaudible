# core/preprocessing/region_mask_generator.py
"""
Region mask generator with bit packing.
Generates track_regions.bin in big-endian format
from ISRCs stored in track_index.bin.

Region          |Index|Binary Representation
--------------------------------------------
           Anglo|  0  |  0b000
Western European|  1  |  0b001
Eastern European|  2  |  0b010
        Hispanic|  3  |  0b011
           Asian|  4  |  0b100
          Indian|  5  |  0b101
  Middle Eastern|  6  |  0b110
           Other|  7  |  0b111

Format (example):

Hex         0x05     0x39     0x77   |   0x92     0x49     0x24

Binary    00000101 00111001 01110111 | 10010010 01001001 00100100 
          \_/\_/\__/\_/\_/\__/\_/\_/   \_/\_/\__/\_/\_/\__/\_/\_/
Index of   0  1   2  3  4   5  6  7     4  4   4  4  4   4  4  4
Region

"""

import os
import mmap
import numpy as np
from pathlib import Path
from tqdm import tqdm
from .progress import ProgressTracker

class RegionMaskGenerator:
    """Generates region mask file from track_index.bin
    with big-endian bit packing"""
    
    # Region dictionary: 8 geographical/cultural/linguistic regions
    REGION_MAPPING = {
        0: ["AU", "CA", "CB", "GB", "GG", "GX", "IE", "IM", "JE", "NZ", 
            "QM", "QT", "QZ", "UK", "US"],  # Anglo
        1: ["AD", "AT", "BE", "CH", "DE", "DK", "EE", "FI", "FO", "FR", 
            "FX", "GI", "GL", "IS", "IT", "LI", "LU", "MC", "MT", "NL", 
            "NO", "PT", "SE", "SM"],  # Western European
        2: ["AL", "BA", "BG", "BY", "CS", "CY", "CZ", "GR", "HR", "HU", 
            "LT", "LV", "MD", "ME", "MK", "PL", "RO", "RS", "RU", "SI", 
            "SK", "UA", "XK", "YU"],  # Eastern European
        3: ["AR", "BC", "BK", "BO", "BP", "BR", "BX", "BZ", "CL", "CO", 
            "CR", "CU", "DO", "EC", "ES", "GT", "HN", "MX", "NI", "PA", 
            "PE", "PR", "PY", "SV", "UY", "VE"],  # Hispanic
        4: ["BN", "CN", "HK", "ID", "JP", "KG", "KH", "KR", "KS", "KZ", 
            "LA", "MM", "MN", "MO", "MY", "PG", "PH", "SG", "TH", "TL", 
            "TW", "UZ", "VN"],  # Asian
        5: ["BD", "BT", "IN", "LK", "MV", "NP", "PK"],  # Indian
        6: ["AE", "AF", "AM", "AZ", "BH", "DZ", "EG", "GE", "IL", "IQ", 
            "IR", "JO", "KW", "LB", "MA", "OM", "PS", "QA", "SA", "SY", 
            "TN", "TR", "YE"],  # Middle Eastern
        7: ["AG", "AI", "AO", "AW", "BB", "BF", "BI", "BJ", "BM", "BS", 
            "BW", "CD", "CF", "CG", "CI", "CM", "CP", "CV", "CW", "DG", 
            "DM", "ET", "FJ", "GA", "GD", "GH", "GM", "GN", "GQ", "GY", 
            "HT", "JM", "KE", "KM", "KN", "KY", "LC", "LR", "LS", "MF", 
            "MG", "ML", "MP", "MR", "MS", "MU", "MW", "MZ", "NA", "NE", 
            "NG", "PF", "QN", "RW", "SB", "SC", "SD", "SL", "SN", "SO", 
            "SR", "SS", "ST", "SX", "SZ", "TC", "TD", "TG", "TO", "TT", 
            "TZ", "UG", "VC", "VG", "VU", "VV", "ZA", "ZB", "ZM", "ZW", 
            "ZZ"]  # Other
    }
    
    # Reverse lookup for country codes
    COUNTRY_TO_REGION = {}
    for region_id, countries in REGION_MAPPING.items():
        for country in countries:
            COUNTRY_TO_REGION[country] = region_id
    
    INDEX_ENTRY_SIZE = 42  # ISRC(12) + TrackID(22) + Offset(8)
    ISRC_SIZE = 12
    BITS_PER_REGION = 3
    BYTES_PER_8_TRACKS = 3  # 8 tracks * 3 bits = 24 bits = 3 bytes

    def __init__(self, index_path: Path, region_path: Path):
        self.index_path = index_path
        self.region_path = region_path
    
    def generate(self):
        """Generate packed region mask file from index file."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        
        file_size = self.index_path.stat().st_size
        total_entries = file_size // self.INDEX_ENTRY_SIZE
        
        print(f"🔍 Found index file: {self.index_path}")
        print(f"   File size: {file_size / (1024**3):.1f} GB")
        print(f"   Total entries: {total_entries:,}")
        
        # Create region directory if needed
        self.region_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n⚙️  Generating packed region file: {self.region_path}")
        print(f"   Using bit packing: 3 bits per track, 8 tracks per 3 bytes")
        
        try:
            with open(self.index_path, 'rb') as index_file, \
                 open(self.region_path, 'wb') as region_file:
                
                # Memory map index file for faster access
                index_mmap = mmap.mmap(index_file.fileno(), 0, access=mmap.ACCESS_READ)
                
                progress_bar = tqdm(total=total_entries, unit='entries', unit_scale=True)
                
                # Buffer for current region batch
                region_buffer = []
                
                for i in range(total_entries):
                    # Extract ISRC (first 12 bytes of entry)
                    entry_start = i * self.INDEX_ENTRY_SIZE
                    isrc_bytes = index_mmap[entry_start:entry_start + self.ISRC_SIZE]
                    isrc = isrc_bytes.decode('ascii', 'ignore').strip('\x00')
                    
                    # Get region index
                    region_byte = self._get_region_byte(isrc)
                    region_buffer.append(region_byte)
                    
                    # Pack every 8 regions
                    if len(region_buffer) == 8:
                        packed_bytes = self._pack_regions(region_buffer)
                        region_file.write(packed_bytes)
                        region_buffer = []
                    
                    # Update progress
                    if i % 100000 == 0:
                        progress_bar.update(100000)
                
                # Process any remaining regions
                if region_buffer:
                    # Pad with "Other" regions (7) if needed
                    while len(region_buffer) < 8:
                        region_buffer.append(7)
                    packed_bytes = self._pack_regions(region_buffer)
                    region_file.write(packed_bytes)
                
                progress_bar.close()
                index_mmap.close()
            
            # Verify file size
            expected_size = (total_entries + 7) // 8 * self.BYTES_PER_8_TRACKS
            actual_size = self.region_path.stat().st_size
            if actual_size != expected_size:
                print(f"⚠️  File size mismatch: expected {expected_size:,} bytes, got {actual_size:,}")
                return False
            else:
                print(f"✅ Successfully generated packed region file: {self.region_path}")
                print(f"   File size: {actual_size / (1024**3):.1f} GB")
                print(f"   Space savings: {(total_entries - actual_size) / total_entries * 100:.1f}%")
                return True
                
        except Exception as e:
            print(f"💥 Error during region generation: {e}")
            return False
    
    def _get_region_byte(self, isrc: str) -> int:
        """Convert ISRC to region index (0-7)."""
        if len(isrc) < 2:
            return 7  # Default to "Other" region
        
        country_code = isrc[:2].upper()
        return self.COUNTRY_TO_REGION.get(country_code, 7)  # 7 = Other region

    def _pack_regions(self, regions: list) -> bytes:
        """Pack 8 region indices (3 bits each) into 3 bytes (big-endian)"""
        if len(regions) != 8:
            raise ValueError("Must provide exactly 8 regions to pack")
        
        # Combine all bits into a single 24-bit value
        packed = 0
        for i, region in enumerate(regions):
            # Validate region is 0-7 (3 bits)
            if region < 0 or region > 7:
                region = 7
            
            # Shift region into position (MSB first: first region in highest bits)
            packed |= (region << (21 - i * 3))
        
        # Convert to 3 bytes (big-endian)
        return packed.to_bytes(3, 'big')

# For testing
if __name__ == "__main__":
    generator = RegionMaskGenerator(
        index_path=Path("data/vectors/track_index.bin"),
        region_path=Path("data/vectors/track_regions.bin")
    )
    success = generator.generate()
    if success:
        print("✅ Region file generation successful!")
    else:
        print("❌ Region file generation failed")
