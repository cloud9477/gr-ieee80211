/*
 *
 *     GNU Radio IEEE 802.11a/g/n/ac 20M bw and upto 2x2
 *     QAM modulation
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
#include "modulation_impl.h"

namespace gr {
  namespace ieee80211 {

    modulation::sptr
    modulation::make()
    {
      return gnuradio::make_block_sptr<modulation_impl>(
        );
    }


    /*
     * The private constructor
     */
    modulation_impl::modulation_impl()
      : gr::block("modulation",
              gr::io_signature::make(1, 1, sizeof(uint8_t)),
              gr::io_signature::make(1, 1, sizeof(gr_complex)))
    {
      d_sModul = MODUL_S_RD_TAG;
      d_debug = false;
      // prepare training fields
      memset((uint8_t*)d_sigl, 0, sizeof(gr_complex) * 64);
      memset((uint8_t*)d_signl, 0, sizeof(gr_complex) * 384);
      // non legacy stf
      memcpy(d_signl+192, C8P_STF_F, sizeof(gr_complex) * 64);
      // non legacy ltf
      memcpy(d_signl+256, C8P_LTF_NL_F, sizeof(gr_complex) * 64);
      gr_complex tmpPilotL[4] = {gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f)};
      gr_complex tmpPilotNL[4] = {gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f)};
      for(int i=0;i<1408;i++)
      {
        for(int j=0;j<4;j++)
        {
          d_pilotsL[i][j] = tmpPilotL[j] * PILOT_P[i%127];
          d_pilotsHT[i][j] = tmpPilotNL[j] * PILOT_P[(i+3)%127];
          d_pilotsVHT[i][j] = tmpPilotNL[j] * PILOT_P[(i+4)%127];
        }
        gr_complex tmpPilot;
        tmpPilot = tmpPilotNL[0];
        tmpPilotNL[0] = tmpPilotNL[1];
        tmpPilotNL[1] = tmpPilotNL[2];
        tmpPilotNL[2] = tmpPilotNL[3];
        tmpPilotNL[3] = tmpPilot;
      }
    }

    /*
     * Our virtual destructor.
     */
    modulation_impl::~modulation_impl()
    {
    }

    void
    modulation_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      ninput_items_required[0] = noutput_items;
    }

    int
    modulation_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const uint8_t* inChips0 = static_cast<const uint8_t*>(input_items[0]);
      gr_complex* outSig0 = static_cast<gr_complex*>(output_items[0]);
      d_nProc = ninput_items[0];
      d_nGen = noutput_items;
      d_nProced = 0;
      d_nGened = 0;

      if(d_sModul == MODUL_S_RD_TAG)
      {
        get_tags_in_range(d_tags, 0, nitems_read(0) , nitems_read(0) + 1);
        if (d_tags.size())
        {
          pmt::pmt_t d_meta = pmt::make_dict();
          for (auto tag : d_tags){
            d_meta = pmt::dict_add(d_meta, tag.key, tag.value);
          }
          d_pktFormat = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("format"), pmt::from_long(-1)));
          d_pktMcs0 = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("mcs0"), pmt::from_long(-1)));
          d_pktNss0 = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("nss0"), pmt::from_long(-1)));
          d_pktLen0 = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("len0"), pmt::from_long(-1)));
          d_pktSeq = pmt::to_long(pmt::dict_ref(d_meta, pmt::mp("seq"), pmt::from_long(-1)));
          d_sigBitsIntedL = pmt::u8vector_elements(pmt::dict_ref(d_meta, pmt::mp("sigl"), pmt::PMT_NIL));
          gr_complex tmpSigPilots[4] = {gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(1.0f, 0.0f), gr_complex(-1.0f, 0.0f)};
          pmt::pmt_t dict = pmt::make_dict();
          dict = pmt::dict_add(dict, pmt::mp("format"), pmt::from_long(d_pktFormat));
          if(d_pktFormat == C8P_F_VHT_MU || d_pktNss0 > 1)
          {
            // go to clean
          }
          else
          {
            std::cout<<"ieee80211 mod, su #"<<d_pktSeq<<", format:"<<d_pktFormat<<", mcs:"<<d_pktMcs0<<", nss:"<<d_pktNss0<<", len:"<<d_pktLen0<<std::endl;
            formatToModSu(&d_m, d_pktFormat, d_pktMcs0, d_pktNss0, d_pktLen0);
            d_nSymCopied = 0;
            d_nSampSigCopied = 0;
            if(d_pktFormat == C8P_F_VHT)
            {
              d_sigBitsIntedNL = pmt::u8vector_elements(pmt::dict_ref(d_meta, pmt::mp("signl"), pmt::PMT_NIL));
              d_sigBitsIntedB0 = pmt::u8vector_elements(pmt::dict_ref(d_meta, pmt::mp("sigb0"), pmt::PMT_NIL));
              procChipsToQamNonShiftedScL(&d_sigBitsIntedL[0], d_signl, C8P_QAM_BPSK);
              procChipsToQamNonShiftedScL(&d_sigBitsIntedNL[0], d_signl+64, C8P_QAM_BPSK);
              procChipsToQamNonShiftedScL(&d_sigBitsIntedNL[48], d_signl+128, C8P_QAM_QBPSK);
              procChipsToQamNonShiftedScNL(&d_sigBitsIntedB0[0], d_signl+320, C8P_QAM_BPSK);
              procInsertPilots(d_signl, tmpSigPilots);
              procInsertPilots(d_signl+64, tmpSigPilots);
              procInsertPilots(d_signl+128, tmpSigPilots);
              procInsertPilots(d_signl+320, tmpSigPilots);
              d_sigP0 = d_signl;
              d_nSampSigTotal = 384;
              dict = pmt::dict_add(dict, pmt::mp("packet_len"), pmt::from_long(6+d_m.nSym+MODUL_N_PADSYM));
            }
            else if(d_pktFormat == C8P_F_HT)
            {
              d_sigBitsIntedNL = pmt::u8vector_elements(pmt::dict_ref(d_meta, pmt::mp("signl"), pmt::PMT_NIL));
              procChipsToQamNonShiftedScL(&d_sigBitsIntedL[0], d_signl, C8P_QAM_BPSK);
              procChipsToQamNonShiftedScL(&d_sigBitsIntedNL[0], d_signl+64, C8P_QAM_QBPSK);
              procChipsToQamNonShiftedScL(&d_sigBitsIntedNL[48], d_signl+128, C8P_QAM_QBPSK);
              procInsertPilots(d_signl, tmpSigPilots);
              procInsertPilots(d_signl+64, tmpSigPilots);
              procInsertPilots(d_signl+128, tmpSigPilots);
              d_sigP0 = d_signl;
              d_nSampSigTotal = 320;
              dict = pmt::dict_add(dict, pmt::mp("packet_len"), pmt::from_long(5+d_m.nSym+MODUL_N_PADSYM));
            }
            else
            {
              procChipsToQamNonShiftedScL(&d_sigBitsIntedL[0], d_sigl, C8P_QAM_BPSK);
              procInsertPilots(d_sigl, tmpSigPilots);
              d_sigP0 = d_sigl;
              d_nSampSigTotal = 64;
              dict = pmt::dict_add(dict, pmt::mp("packet_len"), pmt::from_long(1+d_m.nSym+MODUL_N_PADSYM));
            }
            dict = pmt::dict_add(dict, pmt::mp("nss"), pmt::from_long(d_pktNss0));
          }
          pmt::pmt_t pairs = pmt::dict_items(dict);
          for (size_t i = 0; i < pmt::length(pairs); i++) {
              pmt::pmt_t pair = pmt::nth(i, pairs);
              add_item_tag(0,                   // output port index
                            nitems_written(0),  // output sample index
                            pmt::car(pair),     
                            pmt::cdr(pair),
                            alias_pmt());
          }
          d_sModul = MODUL_S_SIG;
        }
      }

      if(d_sModul == MODUL_S_SIG)
      {
        if(d_nGen < (d_nSampSigTotal - d_nSampSigCopied))
        {
          memcpy(outSig0, d_sigP0 + d_nSampSigCopied, d_nGen * sizeof(gr_complex));
          d_nGened += d_nGen;
          d_nSampSigCopied += d_nGen;
        }
        else
        {
          memcpy(outSig0, d_sigP0 + d_nSampSigCopied, (d_nSampSigTotal - d_nSampSigCopied) * sizeof(gr_complex));
          d_nGened += (d_nSampSigTotal - d_nSampSigCopied);
          d_sModul = MODUL_S_DATA;
        }
      }

      if(d_sModul == MODUL_S_DATA)
      {
        while(true)
        {
          if(d_nSymCopied < (d_m.nSym+MODUL_N_PADSYM))
          {
            if(d_nSymCopied >= d_m.nSym && ((d_nGen - d_nGened) >= 64))
            {
              memset((uint8_t*)(outSig0 + d_nGened), 0, sizeof(gr_complex) * 64);
              d_nSymCopied++;
              d_nGened+=64;
            }
            else if((d_nGen - d_nGened) >= 64 && (d_nProc - d_nProced) >= d_m.nSD)
            {
              if(d_m.format == C8P_F_L)
              {
                procChipsToQamNonShiftedScL(inChips0 + d_nProced, outSig0 + d_nGened, d_m.mod);
                procInsertPilots(outSig0 + d_nGened, d_pilotsL[d_nSymCopied]);
                memset((uint8_t*)(outSig0 + d_nGened), 0, sizeof(gr_complex) * 6);
                memset((uint8_t*)(outSig0 + d_nGened + 59), 0, sizeof(gr_complex) * 5);
              }
              else if(d_m.format == C8P_F_VHT)
              {
                procChipsToQamNonShiftedScNL(inChips0 + d_nProced, outSig0 + d_nGened, d_m.mod);
                procInsertPilots(outSig0 + d_nGened, d_pilotsVHT[d_nSymCopied]);
              }
              else
              {
                procChipsToQamNonShiftedScNL(inChips0 + d_nProced, outSig0 + d_nGened, d_m.mod);
                procInsertPilots(outSig0 + d_nGened, d_pilotsHT[d_nSymCopied]);
              }
              d_nSymCopied++;
              d_nProced+=d_m.nSD;
              d_nGened+=64;
            }
            else
            {
              break;
            }
          }
          else
          {
            d_sModul = MODUL_S_CLEAN;
            break;
          }
        }
      }

      if(d_sModul == MODUL_S_CLEAN)
      {
        if((d_nProc - d_nProced) >= MODUL_GR_GAP)
        {
          d_nProced += MODUL_GR_GAP;
          d_sModul = MODUL_S_RD_TAG;
        }
      }

      consume_each (d_nProced);
      return d_nGened;
    }

  } /* namespace ieee80211 */
} /* namespace gr */
