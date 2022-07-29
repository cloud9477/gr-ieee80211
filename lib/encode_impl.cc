/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 2x2
 *     Encoder of 802.11a/g/n/ac 1x1 and 2x2 payload part
 *     Copyright (C) June 1, 2022  Zelin Yun
 *
 *     This program is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU Affero General Public License as published
 *     by the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     This program is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU Affero General Public License for more details.
 *
 *     You should have received a copy of the GNU Affero General Public License
 *     along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <gnuradio/io_signature.h>
#include "encode_impl.h"

uint8_t test_legacyPkt[105] = {
  0, 7, 1, 100, 0,     // format 0, mcs 0, nss 1, len 100
  8, 1, 48, 0, 102, 85, 68, 51, 34, 1, 102, 85, 68, 51, 34, 2, 102, 85, 68, 51, 
  34, 1, 192, 2, 170, 170, 3, 0, 0, 0, 8, 0, 69, 0, 0, 64, 241, 161, 64, 0, 64, 
  17, 173, 182, 192, 168, 13, 3, 192, 168, 13, 1, 140, 72, 34, 185, 0, 44, 22, 
  43, 109, 122, 108, 116, 98, 111, 121, 66, 48, 55, 110, 110, 77, 114, 51, 110, 
  79, 121, 121, 99, 73, 105, 115, 55, 114, 114, 78, 75, 90, 122, 112, 117, 105, 
  78, 79, 116, 251, 174, 122, 183
};

uint8_t test_htPkt[105] = {
  1, 15, 2, 100, 0,
  8, 1, 48, 0, 102, 85, 68, 51, 34, 1, 102, 85, 68, 51, 34, 2, 102, 85, 68, 51, 
  34, 1, 192, 2, 170, 170, 3, 0, 0, 0, 8, 0, 69, 0, 0, 64, 241, 161, 64, 0, 64, 
  17, 173, 182, 192, 168, 13, 3, 192, 168, 13, 1, 140, 72, 34, 185, 0, 44, 22, 
  43, 109, 122, 108, 116, 98, 111, 121, 66, 48, 55, 110, 110, 77, 114, 51, 110, 
  79, 121, 121, 99, 73, 105, 115, 55, 114, 114, 78, 75, 90, 122, 112, 117, 105, 
  78, 79, 116, 251, 174, 122, 183
};

uint8_t test_vhtPkt[105] = {
  2, 0, 2, 100, 0,
  1, 6, 157, 78, 136, 1, 110, 0, 244, 105, 213, 128, 15, 160, 0, 192, 202, 177, 
  91, 225, 244, 105, 213, 128, 15, 160, 0, 169, 0, 0, 170, 170, 3, 0, 0, 0, 8, 
  0, 69, 0, 0, 58, 171, 2, 64, 0, 64, 17, 123, 150, 10, 10, 0, 6, 10, 10, 0, 1, 
  153, 211, 34, 185, 0, 38, 16, 236, 49, 50, 51, 52, 53, 54, 55, 56, 57, 48, 49, 
  50, 51, 52, 53, 54, 55, 56, 57, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 48, 
  41, 169, 161, 121
};

uint8_t test_vhtPktNdp[5] = {
  // format 2, mcs 0, nss 2, len 0 0
  2, 0, 2, 0, 0
};

uint8_t test_bfQbytesR[1024] = {
  0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 25, 63, 68, 63, 96, 99, 47, 62, 169, 97, 232, 62, 181, 191, 108, 191, 132, 226, 67, 63, 81, 245, 52, 62, 5, 166, 225, 62, 73, 178, 106, 191, 144, 192, 64, 63, 170, 154, 57, 62, 252, 218, 226, 62, 142, 98, 107, 191, 215, 232, 66, 63, 163, 181, 55, 62, 71, 167, 232, 62, 122, 223, 107, 191, 195, 183, 67, 63, 119, 166, 66, 62, 32, 82, 212, 62, 80, 140, 105, 191, 52, 119, 63, 63, 152, 16, 76, 62, 141, 62, 219, 62, 144, 136, 104, 191, 212, 234, 65, 63, 89, 78, 56, 62, 52, 62, 226, 62, 38, 138, 106, 191, 187, 202, 65, 63, 128, 115, 52, 62, 212, 60, 223, 62, 172, 76, 107, 191, 79, 170, 65, 63, 120, 160, 48, 62, 28, 45, 220, 62, 69, 8, 108, 191, 113, 2, 63, 63, 241, 116, 89, 62, 8, 39, 228, 62, 239, 101, 103, 191, 89, 160, 63, 63, 79, 109, 37, 62, 231, 142, 231, 62, 15, 76, 107, 191, 252, 124, 64, 63, 80, 65, 39, 62, 180, 89, 235, 62, 42, 35, 108, 191, 59, 69, 65, 63, 81, 51, 53, 62, 72, 126, 233, 62, 94, 146, 106, 191, 47, 146, 65, 63, 82, 170, 62, 62, 193, 118, 226, 62, 161, 215, 106, 191, 24, 145, 65, 63, 127, 79, 76, 62, 205, 48, 225, 62, 117, 147, 106, 191, 9, 251, 72, 63, 43, 8, 41, 62, 17, 154, 216, 62, 36, 61, 107, 191, 24, 141, 65, 63, 33, 17, 75, 62, 29, 102, 223, 62, 42, 216, 103, 191, 169, 253, 65, 63, 109, 183, 51, 62, 4, 165, 230, 62, 122, 159, 106, 191, 53, 133, 63, 63, 132, 206, 71, 62, 130, 41, 230, 62, 174, 48, 105, 191, 212, 190, 64, 63, 99, 44, 59, 62, 66, 203, 228, 62, 32, 211, 107, 191, 162, 66, 66, 63, 165, 14, 67, 62, 93, 46, 222, 62, 34, 66, 105, 191, 253, 65, 66, 63, 144, 135, 60, 62, 6, 16, 225, 62, 104, 35, 107, 191, 69, 67, 66, 63, 70, 12, 54, 62, 16, 191, 227, 62, 15, 226, 108, 191, 145, 23, 65, 63, 47, 221, 66, 62, 182, 193, 221, 62, 111, 151, 105, 191, 36, 173, 62, 63, 72, 243, 59, 62, 16, 69, 230, 62, 36, 41, 104, 191, 129, 154, 63, 63, 65, 120, 72, 62, 236, 224, 228, 62, 14, 190, 106, 191, 96, 159, 68, 63, 15, 62, 66, 62, 29, 148, 217, 62, 166, 215, 104, 191, 230, 70, 66, 63, 66, 44, 69, 62, 182, 247, 229, 62, 82, 109, 105, 191, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 63, 60, 61, 63, 150, 214, 67, 62, 201, 6, 231, 62, 138, 239, 104, 191, 115, 121, 60, 63, 175, 244, 62, 62, 101, 107, 231, 62, 93, 94, 104, 191, 60, 125, 64, 63, 37, 158, 68, 62, 92, 209, 224, 62, 7, 54, 106, 191, 240, 40, 66, 63, 111, 89, 38, 62, 92, 46, 227, 62, 164, 157, 106, 191, 87, 45, 67, 63, 192, 245, 49, 62, 142, 112, 229, 62, 230, 180, 105, 191, 74, 235, 68, 63, 114, 41, 63, 62, 241, 37, 223, 62, 106, 250, 105, 191, 202, 104, 65, 63, 138, 252, 67, 62, 245, 230, 226, 62, 113, 167, 105, 191, 247, 216, 61, 63, 61, 221, 72, 62, 1, 113, 230, 62, 216, 78, 105, 191, 193, 205, 63, 63, 132, 121, 66, 62, 8, 19, 226, 62, 73, 188, 105, 191, 6, 175, 64, 63, 69, 213, 38, 62, 177, 241, 233, 62, 58, 147, 107, 191, 196, 84, 62, 63, 37, 44, 55, 62, 57, 237, 232, 62, 163, 91, 105, 191, 215, 90, 64, 63, 163, 31, 74, 62, 128, 228, 225, 62, 15, 144, 104, 191, 254, 237, 63, 63, 186, 147, 67, 62, 107, 165, 233, 62, 229, 122, 106, 191, 241, 142, 62, 63, 3, 109, 62, 62, 222, 119, 231, 62, 154, 26, 104, 191, 40, 230, 61, 63, 194, 49, 71, 62, 97, 66, 226, 62, 141, 240, 104, 191, 107, 205, 68, 63, 241, 234, 56, 62, 41, 137, 215, 62, 60, 182, 107, 191, 100, 51, 63, 63, 144, 103, 67, 62, 61, 83, 236, 62, 219, 123, 106, 191, 202, 28, 67, 63, 216, 14, 60, 62, 35, 87, 214, 62, 32, 179, 105, 191, 10, 39, 59, 63, 4, 11, 80, 62, 196, 30, 235, 62, 91, 201, 103, 191, 22, 158, 64, 63, 153, 231, 77, 62, 132, 200, 232, 62, 174, 183, 105, 191, 44, 185, 63, 63, 64, 131, 66, 62, 25, 103, 233, 62, 103, 127, 106, 191, 60, 215, 62, 63, 5, 14, 55, 62, 42, 228, 233, 62, 35, 66, 107, 191, 58, 50, 60, 63, 107, 174, 80, 62, 204, 102, 233, 62, 46, 48, 105, 191, 247, 113, 63, 63, 18, 112, 60, 62, 217, 128, 227, 62, 236, 236, 103, 191, 254, 190, 66, 63, 26, 78, 40, 62, 3, 222, 226, 62, 235, 123, 108, 191, 59, 66, 59, 63, 9, 221, 61, 62, 98, 13, 238, 62, 104, 84, 105, 191, 36, 233, 60, 63, 26, 85, 54, 62, 246, 172, 243, 62, 111, 22, 104, 191, 14, 100, 60, 63, 180, 145, 69, 62, 194, 167, 233, 62, 59, 25, 104, 191, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63
};

uint8_t test_bfQbytesI[1024] = {
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 142, 126, 204, 190, 74, 148, 141, 62, 246, 147, 93, 190, 57, 20, 74, 62, 129, 89, 206, 190, 220, 52, 144, 62, 252, 70, 118, 190, 97, 143, 98, 62, 215, 253, 212, 190, 152, 230, 145, 62, 14, 60, 129, 190, 76, 28, 78, 62, 112, 30, 206, 190, 78, 253, 137, 62, 122, 0, 105, 190, 204, 49, 92, 62, 4, 185, 210, 190, 63, 244, 145, 62, 137, 107, 139, 190, 229, 178, 101, 62, 250, 54, 214, 190, 31, 157, 143, 62, 254, 205, 146, 190, 54, 124, 115, 62, 115, 65, 211, 190, 236, 253, 142, 62, 32, 76, 124, 190, 229, 132, 101, 62, 147, 135, 213, 190, 103, 238, 142, 62, 176, 108, 128, 190, 161, 24, 92, 62, 235, 209, 215, 190, 1, 218, 142, 62, 82, 160, 130, 190, 241, 165, 82, 62, 52, 39, 216, 190, 159, 42, 142, 62, 134, 5, 132, 190, 86, 110, 124, 62, 129, 186, 212, 190, 22, 43, 148, 62, 185, 4, 128, 190, 127, 20, 90, 62, 61, 228, 212, 190, 238, 47, 145, 62, 242, 254, 101, 190, 143, 6, 82, 62, 10, 172, 212, 190, 169, 231, 144, 62, 199, 223, 99, 190, 89, 166, 98, 62, 233, 39, 215, 190, 43, 126, 143, 62, 106, 97, 114, 190, 13, 211, 89, 62, 130, 255, 211, 190, 195, 16, 138, 62, 198, 232, 128, 190, 209, 252, 95, 62, 25, 200, 201, 190, 115, 200, 141, 62, 225, 127, 99, 190, 202, 205, 104, 62, 248, 50, 211, 190, 158, 35, 139, 62, 6, 91, 133, 190, 155, 42, 132, 62, 53, 25, 209, 190, 143, 66, 150, 62, 88, 114, 114, 190, 158, 153, 84, 62, 82, 192, 215, 190, 196, 166, 142, 62, 1, 50, 124, 190, 98, 49, 111, 62, 126, 85, 208, 190, 83, 34, 140, 62, 28, 98, 133, 190, 215, 141, 84, 62, 250, 183, 213, 190, 15, 138, 147, 62, 184, 53, 126, 190, 101, 3, 102, 62, 133, 192, 214, 190, 198, 232, 145, 62, 215, 59, 112, 190, 167, 237, 79, 62, 91, 189, 215, 190, 70, 58, 144, 62, 178, 24, 98, 190, 107, 128, 57, 62, 232, 174, 212, 190, 73, 33, 142, 62, 170, 93, 136, 190, 116, 62, 110, 62, 185, 71, 211, 190, 85, 233, 142, 62, 46, 6, 138, 190, 74, 130, 131, 62, 246, 31, 217, 190, 123, 150, 142, 62, 165, 35, 123, 190, 236, 17, 85, 62, 120, 223, 205, 190, 93, 129, 143, 62, 182, 85, 133, 190, 138, 6, 119, 62, 112, 220, 200, 190, 206, 10, 142, 62, 84, 240, 133, 190, 197, 33, 111, 62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34, 56, 213, 190, 209, 37, 141, 62, 148, 169, 141, 190, 160, 198, 121, 62, 98, 138, 215, 190, 187, 140, 143, 62, 215, 144, 141, 190, 183, 56, 128, 62, 55, 79, 216, 190, 189, 57, 147, 62, 21, 218, 128, 190, 15, 102, 85, 62, 8, 68, 210, 190, 176, 118, 146, 62, 162, 58, 121, 190, 117, 52, 105, 62, 130, 118, 201, 190, 185, 8, 145, 62, 90, 153, 128, 190, 146, 159, 114, 62, 140, 150, 203, 190, 32, 242, 140, 62, 150, 85, 123, 190, 150, 252, 109, 62, 194, 169, 210, 190, 225, 89, 143, 62, 177, 10, 129, 190, 162, 93, 105, 62, 37, 153, 217, 190, 18, 183, 145, 62, 77, 96, 132, 190, 188, 219, 100, 62, 132, 210, 213, 190, 3, 98, 142, 62, 14, 194, 134, 190, 118, 175, 107, 62, 186, 168, 206, 190, 211, 218, 145, 62, 72, 139, 126, 190, 237, 113, 90, 62, 255, 202, 215, 190, 214, 211, 149, 62, 209, 34, 128, 190, 162, 61, 104, 62, 20, 45, 212, 190, 138, 83, 145, 62, 195, 133, 134, 190, 50, 147, 112, 62, 189, 179, 201, 190, 150, 198, 133, 62, 102, 184, 139, 190, 55, 255, 114, 62, 206, 2, 211, 190, 72, 72, 150, 62, 110, 20, 137, 190, 250, 252, 116, 62, 211, 140, 207, 190, 160, 51, 135, 62, 12, 182, 153, 190, 93, 1, 130, 62, 153, 167, 207, 190, 127, 161, 142, 62, 205, 211, 132, 190, 12, 223, 81, 62, 190, 235, 201, 190, 185, 187, 139, 62, 210, 228, 138, 190, 123, 71, 101, 62, 198, 195, 215, 190, 112, 255, 142, 62, 210, 205, 131, 190, 216, 238, 111, 62, 57, 56, 216, 190, 249, 208, 143, 62, 131, 114, 141, 190, 232, 235, 122, 62, 73, 223, 209, 190, 149, 55, 139, 62, 208, 6, 121, 190, 230, 209, 105, 62, 3, 154, 209, 190, 3, 115, 141, 62, 119, 61, 129, 190, 115, 146, 97, 62, 61, 79, 209, 190, 113, 120, 143, 62, 253, 247, 133, 190, 103, 58, 89, 62, 248, 106, 215, 190, 16, 208, 142, 62, 91, 248, 139, 190, 210, 31, 103, 62, 89, 137, 214, 190, 189, 188, 147, 62, 193, 63, 133, 190, 73, 53, 127, 62, 139, 181, 205, 190, 3, 24, 142, 62, 195, 13, 129, 190, 84, 103, 83, 62, 108, 20, 219, 190, 11, 58, 149, 62, 143, 53, 131, 190, 234, 215, 100, 62, 26, 43, 207, 190, 169, 144, 146, 62, 208, 194, 130, 190, 217, 248, 129, 62, 109, 142, 213, 190, 230, 27, 139, 62, 92, 88, 141, 190, 31, 127, 132, 62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

// vht mu-mimo packet: format 1B, mcs0 1B, nss0 1B, len0 2B, mcs1 1B, nss1 1B, len1 2B, groupID 1B.

uint8_t test_vhtPktMu[210] = {
  3, 0, 1, 100, 0, 0, 1, 100, 0, 2,

  1, 6, 157, 78, 136, 1, 110, 0, 244, 105, 213, 128, 15, 160, 0, 192, 202, 177, 91, 225, 244, 105, 213, 128, 15, 160, 0, 169, 0, 0, 170, 170, 3, 0, 0, 0, 8, 0, 69, 0, 0, 58, 171, 2, 64, 0, 64, 17, 123, 150, 10, 10, 0, 6, 10, 10, 0, 1, 153, 211, 34, 185, 0, 38, 75, 110, 84, 104, 105, 115, 32, 105, 115, 32, 112, 97, 99, 107, 101, 116, 32, 102, 111, 114, 32, 115, 116, 97, 116, 105, 111, 110, 32, 48, 48, 49, 192, 175, 19, 101,

  1, 6, 157, 78, 136, 1, 110, 0, 244, 105, 213, 128, 15, 160, 0, 192, 202, 177, 91, 225, 244, 105, 213, 128, 15, 160, 0, 169, 0, 0, 170, 170, 3, 0, 0, 0, 8, 0, 69, 0, 0, 58, 171, 2, 64, 0, 64, 17, 123, 150, 10, 10, 0, 6, 10, 10, 0, 1, 153, 211, 34, 185, 0, 38, 75, 109, 84, 104, 105, 115, 32, 105, 115, 32, 112, 97, 99, 107, 101, 116, 32, 102, 111, 114, 32, 115, 116, 97, 116, 105, 111, 110, 32, 48, 48, 50, 9, 199, 50, 239
};

namespace gr {
  namespace ieee80211 {

    encode::sptr
    encode::make(const std::string& tsb_tag_key)
    {
      return gnuradio::make_block_sptr<encode_impl>(tsb_tag_key
        );
    }


    /*
     * The private constructor
     */
    encode_impl::encode_impl(const std::string& tsb_tag_key)
      : gr::tagged_stream_block("encode",
              gr::io_signature::make(0, 0, 0),
              gr::io_signature::make(2, 2, sizeof(uint8_t)), tsb_tag_key)
    {
      //message_port_register_in(pdu::pdu_port_id());
      d_sEncode = ENCODE_S_IDLE;
      d_debug = true;
      d_pktSeq = 0;
      d_nChipsWithPadded = 624;   // sometimes num of chips are too small which do not trigger the stream passing

      // memset(d_vhtBfQbytesR, 0, 1024);
      // memset(d_vhtBfQbytesI, 0, 1024);

      memcpy(d_vhtBfQbytesR, test_bfQbytesR, 1024);
      memcpy(d_vhtBfQbytesI, test_bfQbytesI, 1024);

      message_port_register_in(pmt::mp("pdus"));
      set_msg_handler(pmt::mp("pdus"), boost::bind(&encode_impl::msgRead, this, _1));
    }

    /*
     * Our virtual destructor.
     */
    encode_impl::~encode_impl()
    {
    }

    void
    encode_impl::msgRead(pmt::pmt_t msg)
    {
      /* 1B format, 1B mcs, 1B nss, 2B len, total 5B, len is 0 then NDP*/
      pmt::pmt_t vector = pmt::cdr(msg);
      int tmpMsgLen = pmt::blob_length(vector);
      size_t tmpOffset(0);
      const uint8_t *tmpPkt = (const uint8_t *)pmt::uniform_vector_elements(vector, tmpOffset);
      if((tmpMsgLen < 5) || (tmpMsgLen > DECODE_D_MAX)){
        return;
      }

      // uint8_t* tmpPkt = test_vhtPktMu;

      // if(tmpPkt[0] == C8P_F_VHT_BFQ_R)
      // {
      //   memcpy(d_vhtBfQbytesR, &tmpPkt[1], 1024);
      //   std::cout<<"beamforming Q real updated"<<std::endl;
      //   return;
      // }

      // if(tmpPkt[0] == C8P_F_VHT_BFQ_I)
      // {
      //   memcpy(d_vhtBfQbytesI, &tmpPkt[1], 1024);
      //   std::cout<<"beamforming Q real updated"<<std::endl;
      //   return;
      // }

      int tmpHeaderShift;
      int tmpFormat = (int)tmpPkt[0];
      if(tmpFormat == C8P_F_VHT_MU)
      {
        // byte 0 format, user0 1-4, user1 5-8, group ID 9
        tmpHeaderShift = 10;
        int tmpMcs0 = (int)tmpPkt[1];
        int tmpNss0 = (int)tmpPkt[2];
        int tmpLen0 = ((int)tmpPkt[4] * 256  + (int)tmpPkt[3]);
        int tmpMcs1 = (int)tmpPkt[5];
        int tmpNss1 = (int)tmpPkt[6];
        int tmpLen1 = ((int)tmpPkt[8] * 256  + (int)tmpPkt[7]);
        int tmpGroupId = (int)tmpPkt[9];
        dout<<"ieee80211 encode, new msg, format:"<<tmpFormat<<std::endl;
        dout<<"ieee80211 encode, new msg, mcs0:"<<tmpMcs0<<", nSS0:"<<tmpNss0<<", len0:"<<tmpLen0<<", mcs1:"<<tmpMcs1<<", nSS1:"<<tmpNss1<<", len1:"<<tmpLen1<<std::endl;
        formatToModMu(&d_m, tmpMcs0, 1, tmpLen0, tmpMcs1, 1, tmpLen1);
        d_m.groupId = tmpGroupId;
        // convert bytes to Q
        dout<<"ieee80211 encode, build Q"<<std::endl;
        float* tmpFloatPR = (float*)d_vhtBfQbytesR;
        float* tmpFloatPI = (float*)d_vhtBfQbytesI;
        d_tagBfQ.clear();
        d_tagBfQ.reserve(256);
        gr_complex tmpQValue;
        for(int i=0;i<256;i++)
        {
          tmpQValue = gr_complex(*tmpFloatPR, *tmpFloatPI);
          tmpFloatPR += 1;
          tmpFloatPI += 1;
          // dout<<tmpQValue<<std::endl;
          d_tagBfQ.push_back(tmpQValue);
        }
      }
      else
      {
        tmpHeaderShift = 5;
        int tmpMcs = (int)tmpPkt[1];
        int tmpNss = (int)tmpPkt[2];
        int tmpLen = ((int)tmpPkt[4] * 256  + (int)tmpPkt[3]);
        dout<<"ieee80211 encode, new msg, format:"<<tmpFormat<<", mcs:"<<tmpMcs<<", nSS:"<<tmpNss<<", len:"<<tmpLen<<std::endl;
        formatToModSu(&d_m, tmpFormat, tmpMcs, tmpNss, tmpLen);
      }

      if(d_m.format == C8P_F_L)
      {
        // legacy
        dout<<"ieee80211 encode, legacy packet"<<std::endl;
        legacySigBitsGen(d_legacySig, d_legacySigCoded, d_m.mcs, d_m.len);
        procIntelLegacyBpsk(d_legacySigCoded, d_legacySigInted);

        uint8_t* tmpDataP = d_dataBits;
        memset(tmpDataP, 0, 16);
        tmpDataP += 16;
        for(int i=0;i<d_m.len;i++)
        {
          for(int j=0;j<8;j++)
          {
            tmpDataP[j] = (tmpPkt[i + tmpHeaderShift] >> j) & 0x01;
          }
          tmpDataP += 8;
        }
        // tail
        memset(tmpDataP, 0, 6);
        tmpDataP += 6;
        // pad
        memset(tmpDataP, 0, (d_m.nSym * d_m.nDBPS - 22 - d_m.len*8));
      }
      else if(d_m.format == C8P_F_VHT)
      {
        // vht
        dout<<"ieee80211 encode, vht packet"<<std::endl;
        vhtSigABitsGen(d_vhtSigA, d_vhtSigACoded, &d_m);
        dout<<"ieee80211 encode, vht packet sig a bits"<<std::endl;
        for(int i=0;i<48;i++)
        {
          dout<<(int)d_vhtSigA[i]<<" ";
        }
        dout<<std::endl;
        procIntelLegacyBpsk(&d_vhtSigACoded[0], &d_vhtSigAInted[0]);
        procIntelLegacyBpsk(&d_vhtSigACoded[48], &d_vhtSigAInted[48]);
        if(d_m.sumu)
        {
          vhtSigB20BitsGenMU(d_vhtSigB, d_vhtSigBCoded, d_vhtSigBCrc8, d_vhtSigBMu1, d_vhtSigBMu1Coded, d_vhtSigBMu1Crc8, &d_m);
          procIntelVhtB20(d_vhtSigBCoded, d_vhtSigBInted);
          procIntelVhtB20(d_vhtSigBMu1Coded, d_vhtSigBMu1Inted);
        }
        else
        {
          vhtSigB20BitsGenSU(d_vhtSigB, d_vhtSigBCoded, d_vhtSigBCrc8, &d_m);
          procIntelVhtB20(d_vhtSigBCoded, d_vhtSigBInted);
        }
        

        if(d_m.nSym > 0)
        {
          // legacy training 16, legacy sig 4, vhtsiga 8, vht training 4+4n, vhtsigb, payload, no short GI
          int tmpTxTime = 20 + 8 + 4 + d_m.nLTF * 4 + 4 + d_m.nSym * 4;
          int tmpLegacyLen = ((tmpTxTime - 20) / 4 + (((tmpTxTime - 20) % 4) != 0)) * 3 - 3;
          legacySigBitsGen(d_legacySig, d_legacySigCoded, 0, tmpLegacyLen);
          procIntelLegacyBpsk(d_legacySigCoded, d_legacySigInted);

          if(d_m.sumu)
          {
            // set mod info to be user 0
            vhtModMuToSu(&d_m, 0);
          }
          // init pointer
          uint8_t* tmpDataP = d_dataBits;
          // 7 scrambler init, 1 reserved
          memset(tmpDataP, 0, 8);
          tmpDataP += 8;
          // 8 sig b crc8
          memcpy(tmpDataP, d_vhtSigBCrc8, 8);
          tmpDataP += 8;
          // data
          for(int i=0;i<d_m.len;i++)
          {
            for(int j=0;j<8;j++)
            {
              tmpDataP[j] = (tmpPkt[i + tmpHeaderShift] >> j) & 0x01;
            }
            tmpDataP += 8;
          }
          tmpHeaderShift += d_m.len;

          // general packet with payload
          int tmpPsduLen = (d_m.nSym * d_m.nDBPS - 16 - 6)/8;
          // EOF padding tmp, copy header bits to pad
          memcpy(tmpDataP, &d_dataBits[16], (tmpPsduLen - d_m.len)*8);
          tmpDataP += (tmpPsduLen - d_m.len)*8;
          // tail pading, all 0, includes tail bits, when scrambling, do not scramble tail
          memset(tmpDataP, 0, (d_m.nSym * d_m.nDBPS - tmpPsduLen*8 - 16));

          // dout<<"bits 1"<<std::endl;
          // for(int i=0;i<d_m.nSym;i++)
          // {
          //   for(int j=0;j<d_m.nDBPS;j++)
          //   {
          //     dout<<(int)d_dataBits[i*d_m.nDBPS + j]<<" ";
          //   }
          //   dout<<std::endl;
          // }

          if(d_m.sumu)
          {
            // set mod info to be user 1, this version no other users
            vhtModMuToSu(&d_m, 1);
            // init pointer
            tmpDataP = d_dataBits2;
            // 7 scrambler init, 1 reserved
            memset(tmpDataP, 0, 8);
            tmpDataP += 8;
            // 8 sig b crc8
            memcpy(tmpDataP, d_vhtSigBMu1Crc8, 8);
            tmpDataP += 8;
            // data
            for(int i=0;i<d_m.len;i++)
            {
              for(int j=0;j<8;j++)
              {
                tmpDataP[j] = (tmpPkt[i + tmpHeaderShift] >> j) & 0x01;
              }
              tmpDataP += 8;
            }
            // general packet with payload
            tmpPsduLen = (d_m.nSym * d_m.nDBPS - 16 - 6)/8;
            // EOF padding tmp, copy header bits to pad
            memcpy(tmpDataP, &d_dataBits[16], (tmpPsduLen - d_m.len)*8);
            tmpDataP += (tmpPsduLen - d_m.len)*8;
            // tail pading, all 0, includes tail bits, when scrambling, do not scramble tail
            memset(tmpDataP, 0, (d_m.nSym * d_m.nDBPS - tmpPsduLen*8 - 16));

            // dout<<"bits 2"<<std::endl;
            // for(int i=0;i<d_m.nSym;i++)
            // {
            //   for(int j=0;j<d_m.nDBPS;j++)
            //   {
            //     dout<<(int)d_dataBits2[i*d_m.nDBPS + j]<<" ";
            //   }
            //   dout<<std::endl;
            // }
          }
        }
        else
        {
          // NDP channel sounding, legacy, vht sig a, vht training, vht sig b
          int tmpTxTime = 20 + 8 + 4 + d_m.nLTF * 4 + 4;
          int tmpLegacyLen = ((tmpTxTime - 20) / 4 + (((tmpTxTime - 20) % 4) != 0)) * 3 - 3;
          legacySigBitsGen(d_legacySig, d_legacySigCoded, 0, tmpLegacyLen);
          procIntelLegacyBpsk(d_legacySigCoded, d_legacySigInted);
        }
      }
      else
      {
        // ht
        dout<<"ieee80211 encode, ht packet"<<std::endl;
        htSigBitsGen(d_htSig, d_htSigCoded, &d_m);
        procIntelLegacyBpsk(&d_htSigCoded[0], &d_htSigInted[0]);
        procIntelLegacyBpsk(&d_htSigCoded[48], &d_htSigInted[48]);
        // legacy training and sig 20, htsig 8, ht training 4+4n, payload, no short GI
        int tmpTxTime = 20 + 8 + 4 + d_m.nLTF * 4 + d_m.nSym * 4;
        int tmpLegacyLen = ((tmpTxTime - 20) / 4 + (((tmpTxTime - 20) % 4) != 0)) * 3 - 3;
        legacySigBitsGen(d_legacySig, d_legacySigCoded, 0, tmpLegacyLen);
        procIntelLegacyBpsk(d_legacySigCoded, d_legacySigInted);

        uint8_t* tmpDataP = d_dataBits;
        // service
        memset(tmpDataP, 0, 16);
        tmpDataP += 16;
        // data
        for(int i=0;i<d_m.len;i++)
        {
          for(int j=0;j<8;j++)
          {
            tmpDataP[j] = (tmpPkt[i + tmpHeaderShift] >> j) & 0x01;
          }
          tmpDataP += 8;
        }
        // tail
        memset(tmpDataP, 0, 6);
        tmpDataP += 6;
        // pad
        memset(tmpDataP, 0, (d_m.nSym * d_m.nDBPS - 22 - d_m.len*8));
      }
      d_sEncode = ENCODE_S_SCEDULE;
    }

    int
    encode_impl::calculate_output_stream_length(const gr_vector_int &ninput_items)
    {
      if(d_sEncode == ENCODE_S_SCEDULE)
      {
        dout<<"ieee80211 encode, schedule in calculate, nSym:"<<d_m.nSym<<", total:"<<(d_m.nSym * d_m.nSD)<<std::endl;
        d_nChipsGen = d_m.nSym * d_m.nSD;
        d_nChipsGenProcd = 0;
        d_sEncode = ENCODE_S_ENCODE;
      }
      if(d_nChipsGen < d_nChipsWithPadded)
      {
        return d_nChipsWithPadded;  // chips with padded
      }
      else
      {
        return d_nChipsGen;
      }
    }

    int
    encode_impl::work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      //auto in = static_cast<const input_type*>(input_items[0]);
      uint8_t* outChips1 = static_cast<uint8_t*>(output_items[0]);
      uint8_t* outChips2 = static_cast<uint8_t*>(output_items[1]);
      d_nGen = noutput_items;

      switch(d_sEncode)
      {
        case ENCODE_S_IDLE:
        {
          return 0;
        }

        case ENCODE_S_SCEDULE:
        {
          return 0;
        }

        case ENCODE_S_ENCODE:
        {
          dout<<"ieee80211 encode, encode and gen tag, seq:"<<d_pktSeq<<std::endl;
          if(d_m.sumu)
          {
            // user 0
            vhtModMuToSu(&d_m, 0);
            scramEncoder(d_dataBits, d_scramBits, (d_m.nSym * d_m.nDBPS - 6), 93);
            memset(&d_scramBits[d_m.nSym * d_m.nDBPS - 6], 0, 6);
            // binary convolutional coding
            bccEncoder(d_scramBits, d_convlBits, d_m.nSym * d_m.nDBPS);
            // puncturing
            punctEncoder(d_convlBits, d_punctBits, d_m.nSym * d_m.nDBPS * 2, &d_m);
            // interleave
            for(int i=0;i<d_m.nSym;i++)
            {
              procInterNonLegacy(&d_punctBits[i*d_m.nCBPS], &d_IntedBits1[i*d_m.nCBPS], &d_m, 0);
            }
            // bits to qam chips
            bitsToChips(d_IntedBits1, d_qamChips1, &d_m);

            // user 1
            vhtModMuToSu(&d_m, 1);
            scramEncoder(d_dataBits2, d_scramBits2, (d_m.nSym * d_m.nDBPS - 6), 93);
            memset(&d_scramBits2[d_m.nSym * d_m.nDBPS - 6], 0, 6);
            // binary convolutional coding
            bccEncoder(d_scramBits2, d_convlBits2, d_m.nSym * d_m.nDBPS);
            // puncturing
            punctEncoder(d_convlBits2, d_punctBits2, d_m.nSym * d_m.nDBPS * 2, &d_m);
            // interleave
            for(int i=0;i<d_m.nSym;i++)
            {
              procInterNonLegacy(&d_punctBits2[i*d_m.nCBPS], &d_IntedBits2[i*d_m.nCBPS], &d_m, 0);
            }
            // bits to qam chips
            bitsToChips(d_IntedBits2, d_qamChips2, &d_m);
          }
          else if(d_m.nSym > 0)
          {
            // scrambling
            if(d_m.format == C8P_F_VHT)
            {
              scramEncoder(d_dataBits, d_scramBits, (d_m.nSym * d_m.nDBPS - 6), 97);
              memset(&d_scramBits[d_m.nSym * d_m.nDBPS - 6], 0, 6);
            }
            else
            {
              scramEncoder(d_dataBits, d_scramBits, (d_m.nSym * d_m.nDBPS), 97);
              memset(&d_scramBits[d_m.len * 8 + 16], 0, 6);
            }
            // binary convolutional coding
            bccEncoder(d_scramBits, d_convlBits, d_m.nSym * d_m.nDBPS);
            // puncturing
            punctEncoder(d_convlBits, d_punctBits, d_m.nSym * d_m.nDBPS * 2, &d_m);
            // interleave and convert to qam chips
            if(d_m.nSS == 1)
            {
              if(d_m.format == C8P_F_L)
              {
                for(int i=0;i<d_m.nSym;i++)
                {
                  procInterLegacy(&d_punctBits[i*d_m.nCBPS], &d_IntedBits1[i*d_m.nCBPS], &d_m);
                }
              }
              else
              {
                for(int i=0;i<d_m.nSym;i++)
                {
                  procInterNonLegacy(&d_punctBits[i*d_m.nCBPS], &d_IntedBits1[i*d_m.nCBPS], &d_m, 0);
                }
              }
              bitsToChips(d_IntedBits1, d_qamChips1, &d_m);
              memset(d_qamChips2, 0, d_m.nSym * d_m.nSD);
            }
            else
            {
              // stream parser first
              streamParser2(d_punctBits, d_parsdBits1, d_parsdBits2, d_m.nSym * d_m.nCBPS, &d_m);
              // interleave
              for(int i=0;i<d_m.nSym;i++)
              {
                procInterNonLegacy(&d_parsdBits1[i*d_m.nCBPSS], &d_IntedBits1[i*d_m.nCBPSS], &d_m, 0);  // iss - 1 = 0
                procInterNonLegacy(&d_parsdBits2[i*d_m.nCBPSS], &d_IntedBits2[i*d_m.nCBPSS], &d_m, 1);  // iss - 1 = 1
              }
              // convert to qam chips
              bitsToChips(d_IntedBits1, d_qamChips1, &d_m);
              bitsToChips(d_IntedBits2, d_qamChips2, &d_m);
            }
          }
          else
          {
            // VHT NDP
          }

          // gen tag
          d_tagLegacyBits.clear();
          d_tagLegacyBits.reserve(48);
          for(int i=0;i<48;i++)
          {
            d_tagLegacyBits.push_back(d_legacySigInted[i]);
          }
          pmt::pmt_t dict = pmt::make_dict();
          if(d_m.sumu)
          {
            dict = pmt::dict_add(dict, pmt::mp("format"), pmt::from_long(C8P_F_VHT_MU));
            dict = pmt::dict_add(dict, pmt::mp("mcs0"), pmt::from_long(d_m.mcsMu[0]));
            dict = pmt::dict_add(dict, pmt::mp("len0"), pmt::from_long(d_m.lenMu[0]));
            dict = pmt::dict_add(dict, pmt::mp("mcs1"), pmt::from_long(d_m.mcsMu[1]));
            dict = pmt::dict_add(dict, pmt::mp("len1"), pmt::from_long(d_m.lenMu[1]));
          }
          else
          {
            dict = pmt::dict_add(dict, pmt::mp("format"), pmt::from_long(d_m.format));
            dict = pmt::dict_add(dict, pmt::mp("mcs"), pmt::from_long(d_m.mcs));
            dict = pmt::dict_add(dict, pmt::mp("nss"), pmt::from_long(d_m.nSS));
            dict = pmt::dict_add(dict, pmt::mp("len"), pmt::from_long(d_m.len));
          }
          dict = pmt::dict_add(dict, pmt::mp("seq"), pmt::from_long(d_pktSeq));
          if(d_nChipsGen < d_nChipsWithPadded)
          {
            dict = pmt::dict_add(dict, pmt::mp("total"), pmt::from_long(d_nChipsWithPadded));   // chips with padded
          }
          else
          {
            dict = pmt::dict_add(dict, pmt::mp("total"), pmt::from_long(d_nChipsGen));
          }
          dict = pmt::dict_add(dict, pmt::mp("lsig"), pmt::init_u8vector(d_tagLegacyBits.size(), d_tagLegacyBits));
          d_pktSeq++;
          if(d_m.format == C8P_F_HT)
          {
            d_tagHtBits.clear();
            d_tagHtBits.reserve(96);
            for(int i=0;i<96;i++)
            {
              d_tagHtBits.push_back(d_htSigInted[i]);
            }
            dict = pmt::dict_add(dict, pmt::mp("htsig"), pmt::init_u8vector(d_tagHtBits.size(), d_tagHtBits));
          }
          else if(d_m.format == C8P_F_VHT)
          {
            // sig a bits
            d_tagVhtABits.clear();
            d_tagVhtABits.reserve(96);
            for(int i=0;i<96;i++)
            {
              d_tagVhtABits.push_back(d_vhtSigAInted[i]);
            }
            dict = pmt::dict_add(dict, pmt::mp("vhtsiga"), pmt::init_u8vector(d_tagVhtABits.size(), d_tagVhtABits));
            // sig b bits
            d_tagVhtBBits.clear();
            d_tagVhtBBits.reserve(52);
            for(int i=0;i<52;i++)
            {
              d_tagVhtBBits.push_back(d_vhtSigBInted[i]);
            }
            dict = pmt::dict_add(dict, pmt::mp("vhtsigb"), pmt::init_u8vector(d_tagVhtBBits.size(), d_tagVhtBBits));
            // mu-mimo, 2nd sig b
            if(d_m.sumu)
            {
              d_tagVhtBMu1Bits.clear();
              d_tagVhtBMu1Bits.reserve(52);
              for(int i=0;i<52;i++)
              {
                d_tagVhtBMu1Bits.push_back(d_vhtSigBMu1Inted[i]);
              }
              dict = pmt::dict_add(dict, pmt::mp("vhtsigb1"), pmt::init_u8vector(d_tagVhtBMu1Bits.size(), d_tagVhtBMu1Bits));
              dict = pmt::dict_add(dict, pmt::mp("vhtbfq"), pmt::init_c32vector(d_tagBfQ.size(), d_tagBfQ));
            }
          }
          pmt::pmt_t pairs = pmt::dict_items(dict);
          for (int i = 0; i < pmt::length(pairs); i++) {
              pmt::pmt_t pair = pmt::nth(i, pairs);
              add_item_tag(0,                   // output port index
                            nitems_written(0),  // output sample index
                            pmt::car(pair),     
                            pmt::cdr(pair),
                            alias_pmt());
          }
          if(d_m.len > 0)
          {
            d_sEncode = ENCODE_S_COPY;
          }
          else
          {
            // NDP, skip copy
            d_sEncode = ENCODE_S_PAD;
          }
          return 0;
        }

        case ENCODE_S_COPY:
        {
          int o1 = 0;
          while((o1 + d_m.nSD) < d_nGen)
          {
            memcpy(&outChips1[o1], &d_qamChips1[d_nChipsGenProcd], d_m.nSD);
            memcpy(&outChips2[o1], &d_qamChips2[d_nChipsGenProcd], d_m.nSD);

            // dout<<"gen: "<<d_nChipsGenProcd<<std::endl;
            // dout<<"ss1: ";
            // for(int j=0;j<d_m.nSD;j++)
            // {
            //   dout<<(int)outChips1[o1+j]<<" ";
            // }
            // dout<<std::endl;
            // dout<<"ss2: ";
            // for(int j=0;j<d_m.nSD;j++)
            // {
            //   dout<<(int)outChips2[o1+j]<<" ";
            // }
            // dout<<std::endl;

            o1 += d_m.nSD;
            d_nChipsGenProcd += d_m.nSD;
            if(d_nChipsGenProcd >= d_nChipsGen)
            {
              dout<<"ieee80211 encode, copy done"<<std::endl;
              if(d_nChipsGen < d_nChipsWithPadded)
              {
                d_sEncode = ENCODE_S_PAD;
              }
              else
              {
                d_sEncode = ENCODE_S_IDLE;
              }
              break;
            }
          }
          return o1;
        }

        case ENCODE_S_PAD:
        {
          if(d_nGen >= (d_nChipsWithPadded - d_nChipsGenProcd))
          {
            memset(outChips1, 0, (d_nChipsWithPadded - d_nChipsGenProcd));
            memset(outChips2, 0, (d_nChipsWithPadded - d_nChipsGenProcd));
            dout<<"ieee80211 encode, padding done"<<std::endl;
            d_sEncode = ENCODE_S_IDLE;
            return (d_nChipsWithPadded - d_nChipsGenProcd);
          }
          else
          {
            memset(outChips1, 0, d_nGen);
            memset(outChips2, 0, d_nGen);
            d_nChipsGenProcd += d_nGen;
          }
          return 0;
        }
      }

      // Tell runtime system how many output items we produced.
      d_sEncode = ENCODE_S_IDLE;
      return 0;
    }

  } /* namespace ieee80211 */
} /* namespace gr */
