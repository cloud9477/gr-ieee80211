/* -*- c++ -*- */
/*
 * Copyright 2022 Zelin Yun.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_IEEE80211_ENCODE_H
#define INCLUDED_IEEE80211_ENCODE_H

#include <gnuradio/ieee80211/api.h>
#include <gnuradio/tagged_stream_block.h>

namespace gr {
  namespace ieee80211 {

    /*!
     * \brief <+description of block+>
     * \ingroup ieee80211
     *
     */
    class IEEE80211_API encode : virtual public gr::tagged_stream_block
    {
     public:
      typedef std::shared_ptr<encode> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of ieee80211::encode.
       *
       * To avoid accidental use of raw pointers, ieee80211::encode's
       * constructor is in a private implementation
       * class. ieee80211::encode::make is the public interface for
       * creating new instances.
       */
      static sptr make(const std::string& lengthtagname = "packet_len");
    };

  } // namespace ieee80211
} // namespace gr

#endif /* INCLUDED_IEEE80211_ENCODE_H */
