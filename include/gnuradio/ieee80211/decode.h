/* -*- c++ -*- */
/*
 * Copyright 2022 Zelin Yun.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_IEEE80211_DECODE_H
#define INCLUDED_IEEE80211_DECODE_H

#include <gnuradio/ieee80211/api.h>
#include <gnuradio/block.h>

namespace gr {
  namespace ieee80211 {

    /*!
     * \brief <+description of block+>
     * \ingroup ieee80211
     *
     */
    class IEEE80211_API decode : virtual public gr::block
    {
     public:
      typedef std::shared_ptr<decode> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of ieee80211::decode.
       *
       * To avoid accidental use of raw pointers, ieee80211::decode's
       * constructor is in a private implementation
       * class. ieee80211::decode::make is the public interface for
       * creating new instances.
       */
      static sptr make();
    };

  } // namespace ieee80211
} // namespace gr

#endif /* INCLUDED_IEEE80211_DECODE_H */
