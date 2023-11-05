/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 20M bw and upto 2x2
 *     QAM modulation
 *     Copyright (C) June 1, 2023  Zelin Yun
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
#include "equalizer2_impl.h"

namespace gr {
  namespace ieee80211 {

    equalizer2::sptr
    equalizer2::make()
    {
      return gnuradio::make_block_sptr<equalizer2_impl>(
        );
    }

    equalizer2_impl::equalizer2_impl()
      : gr::block("equalizer2",
              gr::io_signature::make(2, 2, sizeof(gr_complex)*64),
              gr::io_signature::make(1, 1, sizeof(float)))
    {
      d_HL = std::vector<gr_complex>(53, gr_complex(0.0f, 0.0f));
      d_HNL = std::vector<gr_complex>(57, gr_complex(0.0f, 0.0f));
      d_HNL2 = std::vector<gr_complex>(228, gr_complex(0.0f, 0.0f));
      d_HNL2INV = std::vector<gr_complex>(228, gr_complex(0.0f, 0.0f));
      d_sEq = EQ_S_RDTAG;
    }

    equalizer2_impl::~equalizer2_impl()
    {}

    void
    equalizer2_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      ninput_items_required[0] = noutput_items;
      ninput_items_required[1] = noutput_items;
    }

    int
    equalizer2_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const gr_complex* inSig0 = reinterpret_cast<const gr_complex*>(input_items[0]);
      const gr_complex* inSig1 = reinterpret_cast<const gr_complex*>(input_items[1]);
      float* outLlr = static_cast<float*>(output_items[0]);
      d_nProc = std::min(ninput_items[0], ninput_items[1]);
      d_nGen = noutput_items;
      d_nProced = 0;
      d_nGened = 0;

 

      consume_each (d_nProced);
      return d_nGened;
    }
/*
    void
    equalizer2_impl::htChanUpdate(const gr_complex* sig1, const gr_complex* sig2)
    {
      if(d_m.nSS == 1)
      {
        fftDemod(&sig1[C8P_SYM_SAMP_SHIFT], d_fftLtfOut1);
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35))
          {}
          else
          {
            d_sig1[i] = d_fftLtfOut1[i] / d_H_NL[i][0];
          }
        }
        gr_complex tmpPilotSum = std::conj(d_sig1[7]*d_pilot[2]*PILOT_P[d_pilotP] + d_sig1[21]*d_pilot[3]*PILOT_P[d_pilotP] + d_sig1[43]*d_pilot[0]*PILOT_P[d_pilotP] + d_sig1[57]*d_pilot[1]*PILOT_P[d_pilotP]);
        pilotShift(d_pilot);
        d_pilotP = (d_pilotP + 1) % 127;
        float tmpPilotSumAbs = std::abs(tmpPilotSum);
        int j=26;
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35) || i==7 || i==21 || i==43 || i==57)
          {
          }
          else
          {
            d_qam[0][j] = d_sig1[i] * tmpPilotSum / tmpPilotSumAbs;
            j++;
            if(j >= 52){j = 0;}
          }
        }
      }
      else
      {
        fftDemod(&sig1[C8P_SYM_SAMP_SHIFT], d_fftLtfOut1);
        fftDemod(&sig2[C8P_SYM_SAMP_SHIFT], d_fftLtfOut2);

        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35))
          {}
          else
          {
            gr_complex tmp1 = d_fftLtfOut1[i] * std::conj(d_H_NL[i][0]) + d_fftLtfOut2[i] * std::conj(d_H_NL[i][1]);
            gr_complex tmp2 = d_fftLtfOut1[i] * std::conj(d_H_NL[i][2]) + d_fftLtfOut2[i] * std::conj(d_H_NL[i][3]);
            d_sig1[i] = tmp1 * d_H_NL_INV[i][0] + tmp2 * d_H_NL_INV[i][2];
            d_sig2[i] = tmp1 * d_H_NL_INV[i][1] + tmp2 * d_H_NL_INV[i][3];
          }
        }

        gr_complex tmpPilotSum = std::conj(
          d_sig1[7]*d_pilot[2]*PILOT_P[d_pilotP]*d_pilotNlLtf[2] + 
          d_sig1[21]*d_pilot[3]*PILOT_P[d_pilotP]*d_pilotNlLtf[3] + 
          d_sig1[43]*d_pilot[0]*PILOT_P[d_pilotP]*d_pilotNlLtf[0] + 
          d_sig1[57]*d_pilot[1]*PILOT_P[d_pilotP]*d_pilotNlLtf[1] +
          d_sig2[7]*d_pilot2[2]*PILOT_P[d_pilotP]*d_pilotNlLtf2[2] + 
          d_sig2[21]*d_pilot2[3]*PILOT_P[d_pilotP]*d_pilotNlLtf2[3] + 
          d_sig2[43]*d_pilot2[0]*PILOT_P[d_pilotP]*d_pilotNlLtf2[0] + 
          d_sig2[57]*d_pilot2[1]*PILOT_P[d_pilotP]*d_pilotNlLtf2[1]);

        pilotShift(d_pilot);
        pilotShift(d_pilot2);
        d_pilotP = (d_pilotP + 1) % 127;
        float tmpPilotSumAbs = std::abs(tmpPilotSum);

        int j=26;
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35) || i==7 || i==21 || i==43 || i==57)
          {}
          else
          {
            d_qam[0][j] = d_sig1[i] * tmpPilotSum / tmpPilotSumAbs;
            d_qam[1][j] = d_sig2[i] * tmpPilotSum / tmpPilotSumAbs;
            j++;
            if(j >= 52){j = 0;}
          }
        }
      }
    }

    void
    equalizer2_impl::vhtChanUpdate(const gr_complex* sig1, const gr_complex* sig2)
    {
      if(d_m.nSS == 1)
      {
        fftDemod(&sig1[C8P_SYM_SAMP_SHIFT], d_fftLtfOut1);
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35))
          {}
          else
          {
            d_sig1[i] = d_fftLtfOut1[i] / d_H_NL[i][0];
          }
        }
        gr_complex tmpPilotSum = std::conj(d_sig1[7]*d_pilot[2]*PILOT_P[d_pilotP] + d_sig1[21]*d_pilot[3]*PILOT_P[d_pilotP] + d_sig1[43]*d_pilot[0]*PILOT_P[d_pilotP] + d_sig1[57]*d_pilot[1]*PILOT_P[d_pilotP]);
        pilotShift(d_pilot);
        d_pilotP = (d_pilotP + 1) % 127;
        float tmpPilotSumAbs = std::abs(tmpPilotSum);
        int j=26;
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35) || i==7 || i==21 || i==43 || i==57)
          {
          }
          else
          {
            d_qam[0][j] = d_sig1[i] * tmpPilotSum / tmpPilotSumAbs;
            j++;
            if(j >= 52){j = 0;}
          }
        }
      }
      else
      {
        fftDemod(&sig1[C8P_SYM_SAMP_SHIFT], d_fftLtfOut1);
        fftDemod(&sig2[C8P_SYM_SAMP_SHIFT], d_fftLtfOut2);

        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35))
          {}
          else
          {
            gr_complex tmp1 = d_fftLtfOut1[i] * std::conj(d_H_NL[i][0]) + d_fftLtfOut2[i] * std::conj(d_H_NL[i][1]);
            gr_complex tmp2 = d_fftLtfOut1[i] * std::conj(d_H_NL[i][2]) + d_fftLtfOut2[i] * std::conj(d_H_NL[i][3]);
            d_sig1[i] = tmp1 * d_H_NL_INV[i][0] + tmp2 * d_H_NL_INV[i][2];
            d_sig2[i] = tmp1 * d_H_NL_INV[i][1] + tmp2 * d_H_NL_INV[i][3];
          }
        }
        gr_complex tmpPilotSum = std::conj(
          d_sig1[7]*d_pilot[2]*PILOT_P[d_pilotP]*d_pilotNlLtf[2] + 
          d_sig1[21]*d_pilot[3]*PILOT_P[d_pilotP]*d_pilotNlLtf[3] + 
          d_sig1[43]*d_pilot[0]*PILOT_P[d_pilotP]*d_pilotNlLtf[0] + 
          d_sig1[57]*d_pilot[1]*PILOT_P[d_pilotP]*d_pilotNlLtf[1] +
          d_sig2[7]*d_pilot[2]*PILOT_P[d_pilotP]*d_pilotNlLtf2[2] + 
          d_sig2[21]*d_pilot[3]*PILOT_P[d_pilotP]*d_pilotNlLtf2[3] + 
          d_sig2[43]*d_pilot[0]*PILOT_P[d_pilotP]*d_pilotNlLtf2[0] + 
          d_sig2[57]*d_pilot[1]*PILOT_P[d_pilotP]*d_pilotNlLtf2[1]);
        pilotShift(d_pilot);
        d_pilotP = (d_pilotP + 1) % 127;
        float tmpPilotSumAbs = std::abs(tmpPilotSum);
        int j=26;
        for(int i=0;i<64;i++)
        {
          if(i==0 || (i>=29 && i<=35) || i==7 || i==21 || i==43 || i==57)
          {}
          else
          {
            d_qam[0][j] = d_sig1[i] * tmpPilotSum / tmpPilotSumAbs;
            d_qam[1][j] = d_sig2[i] * tmpPilotSum / tmpPilotSumAbs;
            j++;
            if(j >= 52){j = 0;}
          }
        }

      }
    }

    void
    equalizer2_impl::legacyChanUpdate(const gr_complex* sig1)
    {
      fftDemod(&sig1[C8P_SYM_SAMP_SHIFT], d_fftLtfOut1);
      for(int i=0;i<64;i++)
      {
        if(i==0 || (i>=27 && i<=37))
        {}
        else
        {
          d_sig1[i] = d_fftLtfOut1[i] / d_HL[i];
        }
      }
      gr_complex tmpPilotSum = std::conj(d_sig1[7]*d_pilot[2]*PILOT_P[d_pilotP] + d_sig1[21]*d_pilot[3]*PILOT_P[d_pilotP] + d_sig1[43]*d_pilot[0]*PILOT_P[d_pilotP] + d_sig1[57]*d_pilot[1]*PILOT_P[d_pilotP]);
      d_pilotP = (d_pilotP + 1) % 127;
      float tmpPilotSumAbs = std::abs(tmpPilotSum);
      int j=24;
      for(int i=0;i<64;i++)
      {
        if(i==0 || (i>=27 && i<=37) || i==7 || i==21 || i==43 || i==57)
        {}
        else
        {
          d_qam[0][j] = d_sig1[i] * tmpPilotSum / tmpPilotSumAbs;
          j++;
          if(j >= 48){j = 0;}
        }
      }
    }

    void
    equalizer2_impl::fftDemod(const gr_complex* sig, gr_complex* res)
    {
      memcpy(d_ofdm_fft.get_inbuf(), sig, sizeof(gr_complex)*64);
      d_ofdm_fft.execute();
      memcpy(res, d_ofdm_fft.get_outbuf(), sizeof(gr_complex)*64);
    }
*/
  } /* namespace ieee80211 */
} /* namespace gr */
